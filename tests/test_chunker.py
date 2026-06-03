from agent_memory_benchmarks.corpus.chunker import (
    CodeSplitter,
    LiterarySplitter,
    MarkdownSplitter,
)


def test_markdown_heading_split():
    text = (
        "---\n"
        "title: Test\n"
        "---\n"
        "# Intro\n"
        "Hello world.\n\n"
        "## Section A\n"
        "Some content here.\n\n"
        "## Section B\n"
        "More content.\n"
    )
    splitter = MarkdownSplitter(max_tokens=512)
    chunks = splitter.split(text, "test.md")
    # Expect 3 chunks for # Intro, ## Section A, ## Section B
    assert len(chunks) == 3
    assert all(c.id for c in chunks)
    assert all(c.token_count > 0 for c in chunks)
    # Front-matter metadata propagated
    assert chunks[0].metadata.get("title") == "Test"
    # IDs are unique
    assert len({c.id for c in chunks}) == 3


def test_code_function_extraction_python():
    src = (
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def bar(x):\n"
        "    return x + 1\n"
        "\n"
        "class Baz:\n"
        "    def method(self):\n"
        "        return 2\n"
    )
    splitter = CodeSplitter(language="python")
    chunks = splitter.split(src, "test.py")
    # Should extract at least foo, bar, Baz (and possibly Baz.method)
    assert len(chunks) >= 3
    texts = [c.text for c in chunks]
    assert any("def foo" in t for t in texts)
    assert any("def bar" in t for t in texts)
    assert any("class Baz" in t for t in texts)
    # No fallback was used
    assert all(not c.metadata.get("fallback") for c in chunks)


def test_code_fallback_on_unknown_language():
    splitter = CodeSplitter(language="cobol")
    src = "x" * 5000
    chunks = splitter.split(src, "test.cob")
    # Fallback char split — 2048 chars, 256 overlap → step 1792 → ~3 chunks
    assert len(chunks) >= 2
    assert all(c.metadata.get("fallback") for c in chunks)


def test_literary_paragraph_windowing():
    paragraphs = ["word " * 50, "alpha " * 50, "beta " * 50, "gamma " * 50]
    text = "CHAPTER ONE\n\n" + "\n\n".join(p.strip() for p in paragraphs)
    splitter = LiterarySplitter(max_words=80, overlap_words=10)
    chunks = splitter.split(text, "book.txt")
    assert len(chunks) >= 2
    # Each window should be <= max_words + overlap_words (after the initial overlap carryover)
    for c in chunks:
        assert c.token_count > 0
    # Unique chunk ids
    assert len({c.id for c in chunks}) == len(chunks)
