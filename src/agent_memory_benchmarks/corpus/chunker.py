import hashlib
import re
from dataclasses import dataclass, field
from typing import Protocol

import frontmatter


@dataclass
class Chunk:
    id: str  # sha256(source_path + char_offset)
    text: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

    @staticmethod
    def make_id(source: str, offset: int) -> str:
        return hashlib.sha256(f"{source}:{offset}".encode()).hexdigest()[:16]


class Splitter(Protocol):
    def split(self, text: str, source_path: str) -> list[Chunk]: ...


# Pre-compiled patterns
_HEADING_SPLIT = re.compile(r"\n(?=#{1,3} )")
_SENTENCE_SPLIT = re.compile(r"(?<=\. )|\n\n")
_PARAGRAPH_SPLIT = re.compile(r"\n\n+")
_CHAPTER_HEADING = re.compile(r"^(CHAPTER|PART)\b.*", re.IGNORECASE | re.MULTILINE)


def _word_count(text: str) -> int:
    return len(text.split())


class MarkdownSplitter:
    """Split on H1/H2/H3 boundaries; sub-split overlong sections at max_tokens."""

    def __init__(self, max_tokens: int = 512, overlap: int = 64):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def _sub_split(self, text: str) -> list[str]:
        """Split a long section into pieces of <= max_tokens words at sentence boundaries."""
        words = text.split()
        if len(words) <= self.max_tokens:
            return [text]
        pieces: list[str] = []
        # Try sentence/paragraph boundaries first
        segments = [s for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]
        current: list[str] = []
        current_count = 0
        for seg in segments:
            seg_words = seg.split()
            if current_count + len(seg_words) > self.max_tokens and current:
                pieces.append(" ".join(current).strip())
                # Overlap: keep the last `overlap` words
                if self.overlap > 0:
                    tail_words = " ".join(current).split()[-self.overlap :]
                    current = [" ".join(tail_words)]
                    current_count = len(tail_words)
                else:
                    current = []
                    current_count = 0
            current.append(seg)
            current_count += len(seg_words)
        if current:
            pieces.append(" ".join(current).strip())
        return pieces if pieces else [text]

    def split(self, text: str, source_path: str) -> list[Chunk]:
        post = frontmatter.loads(text)
        body = post.content
        base_metadata: dict = dict(post.metadata) if post.metadata else {}

        # Split on H1/H2/H3 headings while preserving section text
        sections = _HEADING_SPLIT.split(body)
        chunks: list[Chunk] = []
        offset = 0
        for section in sections:
            if not section.strip():
                offset += len(section)
                continue
            pieces = self._sub_split(section)
            piece_offset = offset
            for piece in pieces:
                piece_text = piece.strip()
                if not piece_text:
                    continue
                meta = dict(base_metadata)
                meta["source"] = source_path
                meta["offset"] = piece_offset
                chunks.append(
                    Chunk(
                        id=Chunk.make_id(source_path, piece_offset),
                        text=piece_text,
                        metadata=meta,
                        token_count=_word_count(piece_text),
                    )
                )
                piece_offset += len(piece)
            offset += len(section)
        return chunks


class CodeSplitter:
    """Tree-sitter AST split at function/class boundaries; fallback to char split."""

    # Map language name → (module name, node-type set)
    _LANGUAGES = {
        "python": ("tree_sitter_python", {"function_definition", "class_definition"}),
        "javascript": (
            "tree_sitter_javascript",
            {"function_declaration", "class_declaration", "method_definition"},
        ),
        "typescript": (
            "tree_sitter_typescript",
            {"function_declaration", "class_declaration", "method_definition"},
        ),
        "go": ("tree_sitter_go", {"function_declaration", "method_declaration"}),
        "rust": ("tree_sitter_rust", {"function_item", "impl_item", "struct_item"}),
    }

    _FALLBACK_CHARS = 2048
    _FALLBACK_OVERLAP = 256

    def __init__(self, language: str, max_tokens: int = 512):
        self.language = language
        self.max_tokens = max_tokens

    def _fallback_split(self, text: str, source_path: str) -> list[Chunk]:
        """Fixed-size char split when tree-sitter unavailable or parse fails."""
        chunks: list[Chunk] = []
        step = self._FALLBACK_CHARS - self._FALLBACK_OVERLAP
        if step <= 0:
            step = self._FALLBACK_CHARS
        i = 0
        while i < len(text):
            piece = text[i : i + self._FALLBACK_CHARS]
            if not piece.strip():
                i += step
                continue
            chunks.append(
                Chunk(
                    id=Chunk.make_id(source_path, i),
                    text=piece,
                    metadata={"source": source_path, "offset": i, "fallback": True},
                    token_count=_word_count(piece),
                )
            )
            i += step
        return chunks

    def split(self, text: str, source_path: str) -> list[Chunk]:
        lang_entry = self._LANGUAGES.get(self.language)
        if lang_entry is None:
            return self._fallback_split(text, source_path)

        module_name, node_types = lang_entry
        try:
            import importlib

            from tree_sitter import Language, Parser

            ts_mod = importlib.import_module(module_name)
            # tree-sitter-typescript exposes language_typescript / language_tsx
            if self.language == "typescript":
                lang_callable = getattr(ts_mod, "language_typescript", None) or getattr(ts_mod, "language", None)
            else:
                lang_callable = getattr(ts_mod, "language", None)
            if lang_callable is None:
                return self._fallback_split(text, source_path)
            lang = Language(lang_callable())
            parser = Parser(lang)
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            return self._fallback_split(text, source_path)

        if tree.root_node.has_error:
            # Parse had errors — still try to extract clean nodes; if none, fall back.
            pass

        chunks: list[Chunk] = []
        source_bytes = text.encode("utf-8")

        def walk(node) -> None:
            if node.type in node_types:
                start = node.start_byte
                end = node.end_byte
                snippet = source_bytes[start:end].decode("utf-8", errors="replace")
                if snippet.strip():
                    chunks.append(
                        Chunk(
                            id=Chunk.make_id(source_path, start),
                            text=snippet,
                            metadata={
                                "source": source_path,
                                "offset": start,
                                "node_type": node.type,
                                "language": self.language,
                            },
                            token_count=_word_count(snippet),
                        )
                    )
                # Recurse for nested definitions (e.g., methods inside classes)
            for child in node.children:
                walk(child)

        walk(tree.root_node)

        if not chunks:
            return self._fallback_split(text, source_path)
        return chunks


class LiterarySplitter:
    """Chapter + paragraph-window split."""

    def __init__(self, max_words: int = 200, overlap_words: int = 40):
        self.max_words = max_words
        self.overlap_words = overlap_words

    def _split_chapters(self, text: str) -> list[tuple[int, str]]:
        """Return (offset, chapter_text) pairs. If no chapter headings, returns one entry."""
        matches = list(_CHAPTER_HEADING.finditer(text))
        if not matches:
            return [(0, text)]
        chapters: list[tuple[int, str]] = []
        # Text before first chapter heading
        if matches[0].start() > 0:
            prelude = text[: matches[0].start()]
            if prelude.strip():
                chapters.append((0, prelude))
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chapters.append((start, text[start:end]))
        return chapters

    def _window_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Group paragraphs into max_words windows with overlap_words overlap."""
        if not paragraphs:
            return []
        windows: list[str] = []
        current_words: list[str] = []
        for para in paragraphs:
            words = para.split()
            if not words:
                continue
            if current_words and len(current_words) + len(words) > self.max_words:
                windows.append(" ".join(current_words))
                current_words = current_words[-self.overlap_words :] if self.overlap_words > 0 else []
            current_words.extend(words)
        if current_words:
            windows.append(" ".join(current_words))
        return windows

    def split(self, text: str, source_path: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for chapter_offset, chapter_text in self._split_chapters(text):
            paragraphs = [p for p in _PARAGRAPH_SPLIT.split(chapter_text) if p.strip()]
            windows = self._window_paragraphs(paragraphs)
            window_offset = chapter_offset
            for window in windows:
                meta = {
                    "source": source_path,
                    "offset": window_offset,
                    "chapter_offset": chapter_offset,
                }
                chunks.append(
                    Chunk(
                        id=Chunk.make_id(source_path, window_offset),
                        text=window,
                        metadata=meta,
                        token_count=_word_count(window),
                    )
                )
                # Advance offset by approximate window length so chunk ids stay unique
                window_offset += len(window)
        return chunks
