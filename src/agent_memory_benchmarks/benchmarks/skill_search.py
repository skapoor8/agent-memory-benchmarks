"""Skill Search Benchmark.

Task: Given a task description, retrieve the correct API/tool to invoke
      from a corpus of API descriptions.

Corpora:
  gorilla-apibench:            HuggingFace Transformers subset of gorilla-llm/APIBench
  gorilla-apibench-torchhub:   Torch Hub subset
  gorilla-apibench-tensorhub:  TensorFlow Hub subset
  gorilla-apibench-all:        All three subsets merged

GT:  api_call field from gorilla-llm/APIBench maps instruction → api_name
Metric: nDCG@10, Recall@1
Source: Gorilla LLM APIBench (Patil et al., 2023)
"""

import json
from pathlib import Path

from ..adapters.base import Document
from .common import BenchmarkResult, embed_docs, get_model, run_evaluation

DATA_DIR = Path("data/skill-search")

_PROVIDER_MAP: dict[str, str] = {
    "huggingface": "Hugging Face Transformers",
    "torchhub": "PyTorch",
    "tensorhub": "TensorFlow Hub",
}

# Corpus name → subset key (or "all")
_CORPUS_MAP: dict[str, str] = {
    "gorilla-apibench": "huggingface",  # backward-compatible default
    "gorilla-apibench-huggingface": "huggingface",
    "gorilla-apibench-torchhub": "torchhub",
    "gorilla-apibench-tensorhub": "tensorhub",
    "gorilla-apibench-all": "all",
}

# 30 realistic tool descriptions used as offline fallback so BM25 gets non-zero scores.
# Queries use different but overlapping wording so retrieval is non-trivial.
_FALLBACK_TOOLS = [
    ("file_read", "Read file contents from a path on disk"),
    ("file_write", "Write or overwrite a file on disk with given content"),
    ("web_search", "Search the web and return a list of result URLs and snippets"),
    ("web_fetch", "Fetch the HTML or text content of a URL"),
    ("code_execute", "Execute Python code in a sandboxed interpreter and return stdout"),
    ("sql_query", "Run a SQL SELECT query against a relational database"),
    ("image_generate", "Generate an image from a text prompt using a diffusion model"),
    ("image_caption", "Describe the contents of an image file"),
    ("translate", "Translate text from one language to another"),
    ("summarize", "Summarize a long document into a short paragraph"),
    ("classify", "Classify text into one of a set of predefined categories"),
    ("embed", "Convert text into a dense vector embedding"),
    ("ocr", "Extract text from an image or scanned PDF"),
    ("pdf_extract", "Extract text and metadata from a PDF file"),
    ("email_send", "Send an email to one or more recipients"),
    ("calendar_add", "Create a calendar event with title, time, and attendees"),
    ("slack_post", "Post a message to a Slack channel"),
    ("github_pr", "Open a pull request on a GitHub repository"),
    ("db_insert", "Insert a row into a database table"),
    ("db_update", "Update existing rows in a database table"),
    ("json_parse", "Parse a JSON string and return a Python object"),
    ("csv_read", "Read a CSV file and return rows as a list of dicts"),
    ("zip_extract", "Extract files from a zip archive to a directory"),
    ("http_post", "Send an HTTP POST request with a JSON body"),
    ("http_get", "Send an HTTP GET request and return the response body"),
    ("speech_to_text", "Transcribe audio from a file into text"),
    ("text_to_speech", "Convert text to an audio file using a TTS model"),
    ("vector_search", "Search a vector database for the nearest neighbors to a query"),
    ("rerank", "Rerank a list of documents by relevance to a query"),
    ("chunk_text", "Split a long document into overlapping text chunks"),
]

_FALLBACK_QUERIES = [
    ("file_read", "load the configuration from a yaml file on disk"),
    ("file_write", "save the output report to a local file path"),
    ("web_search", "look up recent papers about transformer architectures"),
    ("web_fetch", "download the HTML from a webpage URL"),
    ("code_execute", "run a Python script and capture the printed output"),
    ("sql_query", "retrieve all rows from a database table matching a condition"),
    ("image_generate", "create a picture of a sunset over mountains from a description"),
    ("image_caption", "describe what is shown in a photograph"),
    ("translate", "convert a sentence from French to English"),
    ("summarize", "condense a long article into a few sentences"),
    ("classify", "label a customer review as positive or negative sentiment"),
    ("embed", "get a vector representation of a sentence for similarity search"),
    ("ocr", "pull the printed text out of a scanned document image"),
    ("pdf_extract", "read the text content from a PDF document"),
    ("email_send", "compose and deliver an email message to a recipient"),
    ("calendar_add", "schedule a meeting for tomorrow at 3pm with two attendees"),
    ("slack_post", "send a notification message to a Slack channel"),
    ("github_pr", "create a pull request to merge a feature branch"),
    ("db_insert", "add a new record to the users table in the database"),
    ("db_update", "modify the status field of an existing database row"),
    ("json_parse", "convert a JSON string into a usable Python dictionary"),
    ("csv_read", "load a spreadsheet file and iterate over its rows"),
    ("zip_extract", "unpack a compressed archive into a folder"),
    ("http_post", "call a REST API endpoint with a JSON payload body"),
    ("http_get", "make a GET request to fetch data from an API URL"),
    ("speech_to_text", "convert a spoken audio recording into written text"),
    ("text_to_speech", "generate an audio file from a written sentence"),
    ("vector_search", "find the most similar items to a query in an embedding database"),
    ("rerank", "sort a list of passages by how relevant they are to a question"),
    ("chunk_text", "break a large document into smaller overlapping segments"),
]


def _synthetic_skillsbench() -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Fallback for offline/CI use. Uses realistic tool descriptions so BM25 scores are non-zero."""
    docs = [Document(id=tool_id, text=desc) for tool_id, desc in _FALLBACK_TOOLS]
    queries = [{"id": f"q_{i}", "text": query_text} for i, (_, query_text) in enumerate(_FALLBACK_QUERIES)]
    qrels = {f"q_{i}": {tool_id} for i, (tool_id, _) in enumerate(_FALLBACK_QUERIES)}
    return docs, queries, qrels


def _load_from_cache(cache: Path) -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    queries: list[dict] = []
    qrels: dict[str, set[str]] = {}
    skill_texts: dict[str, str] = {}
    with cache.open() as f:
        for line in f:
            row = json.loads(line)
            queries.append({"id": row["query_id"], "text": row["query"]})
            qrels[row["query_id"]] = set(row["relevant"])
            for sid in row["relevant"]:
                skill_texts[sid] = row.get("skill_text", sid)
    docs = [Document(id=sid, text=text) for sid, text in skill_texts.items()]
    return docs, queries, qrels


def download_skillsbench(subset: str = "huggingface") -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Load gorilla-llm/APIBench for a single provider subset. Falls back to synthetic if unavailable.

    subset: one of 'huggingface', 'torchhub', 'tensorhub'
    Cached to data/skill-search/qrels-{subset}.jsonl
    """
    if subset not in _PROVIDER_MAP:
        raise ValueError(f"Unknown APIBench subset '{subset}'. Choose from: {list(_PROVIDER_MAP)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Migrate legacy cache (qrels.jsonl → qrels-huggingface.jsonl)
    legacy_cache = DATA_DIR / "qrels.jsonl"
    new_cache = DATA_DIR / f"qrels-{subset}.jsonl"
    if legacy_cache.exists() and not new_cache.exists() and subset == "huggingface":
        legacy_cache.rename(new_cache)

    if new_cache.exists():
        return _load_from_cache(new_cache)

    try:
        from datasets import load_dataset

        ds = load_dataset("gorilla-llm/APIBench", split="train")
        provider_name = _PROVIDER_MAP[subset]

        skill_texts: dict[str, str] = {}
        queries: list[dict] = []
        qrels: dict[str, set[str]] = {}
        seen_instructions: set[str] = set()

        for row in ds:
            if row.get("provider") != provider_name:
                continue
            try:
                raw = row["api_data"]
                api_data = raw if isinstance(raw, dict) else json.loads(raw)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

            api_name = api_data.get("api_name", "").strip()
            description = api_data.get("description", "").strip()
            domain = api_data.get("domain", "").strip()
            if not api_name or not description:
                continue

            if api_name not in skill_texts:
                skill_texts[api_name] = f"{description} {domain}".strip()

            code = row.get("code", "")
            instruction = code.split("###Output:")[0].replace("###Instruction:", "").strip()
            if not instruction or instruction in seen_instructions:
                continue

            seen_instructions.add(instruction)
            q_id = f"q_{len(queries)}"
            queries.append({"id": q_id, "text": instruction})
            qrels[q_id] = {api_name}

        if not queries or not skill_texts:
            return _synthetic_skillsbench()

        with new_cache.open("w") as f:
            for q in queries:
                api_name = next(iter(qrels[q["id"]]))
                f.write(
                    json.dumps(
                        {
                            "query_id": q["id"],
                            "query": q["text"],
                            "relevant": [api_name],
                            "skill_text": skill_texts.get(api_name, api_name),
                        }
                    )
                    + "\n"
                )

        docs = [Document(id=sid, text=text) for sid, text in skill_texts.items()]
        return docs, queries, qrels

    except Exception:
        return _synthetic_skillsbench()


def _load_all_subsets() -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Load and merge all three APIBench subsets.

    Query IDs are namespaced with subset prefix to avoid collisions:
    q_huggingface_0, q_torchhub_0, q_tensorhub_0
    """
    all_docs: dict[str, str] = {}  # api_name → text (dedup across subsets)
    all_queries: list[dict] = []
    all_qrels: dict[str, set[str]] = {}

    for subset in _PROVIDER_MAP:
        docs, queries, qrels = download_skillsbench(subset=subset)
        for doc in docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = doc.text
        for q in queries:
            rel_set = qrels[q["id"]]
            namespaced_id = f"q_{subset}_{q['id'].lstrip('q_')}"
            all_queries.append({"id": namespaced_id, "text": q["text"]})
            all_qrels[namespaced_id] = rel_set

    merged_docs = [Document(id=k, text=v) for k, v in all_docs.items()]
    return merged_docs, all_queries, all_qrels


def run(adapter, corpus: str = "gorilla-apibench", model: str = "small") -> BenchmarkResult:
    if corpus not in _CORPUS_MAP:
        raise ValueError(f"Unknown skill-search corpus '{corpus}'. Choose from: {list(_CORPUS_MAP)}")

    subset_key = _CORPUS_MAP[corpus]
    if subset_key == "all":
        docs, queries, qrels = _load_all_subsets()
    else:
        docs, queries, qrels = download_skillsbench(subset=subset_key)

    docs = embed_docs(docs, model)
    model_obj = get_model(model)
    adapter.index(docs)
    adapter.optimize()

    vecs = model_obj.encode([q["text"] for q in queries]).tolist()
    queries = [{**q, "vector": v} for q, v in zip(queries, vecs, strict=False)]

    result = run_evaluation(adapter, docs, queries, qrels)
    result.benchmark = "skill-search"
    result.corpus = corpus
    result.model = model
    return result
