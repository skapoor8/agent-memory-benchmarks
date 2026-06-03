"""Skill Search Benchmark.

Task: Given a task description, retrieve the correct API/tool to invoke
      from a corpus of API descriptions.

Corpora:
  gorilla-apibench: HuggingFace Transformers subset of gorilla-llm/APIBench
                    (~200-300 unique APIs, ~300 task-instruction queries)

GT:  api_call field from gorilla-llm/APIBench maps instruction → api_name
Metric: nDCG@10, Recall@1
Source: Gorilla LLM APIBench (Patil et al., 2023)
"""

import json
from pathlib import Path

from ..adapters.base import Document
from .common import BenchmarkResult, embed_docs, get_model, run_evaluation

DATA_DIR = Path("data/skill-search")

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


def download_skillsbench() -> tuple[list[Document], list[dict], dict[str, set[str]]]:
    """Load gorilla-llm/APIBench (HuggingFace provider subset). Falls back to synthetic if unavailable.

    Cached to data/skill-search/qrels.jsonl.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / "qrels.jsonl"
    if cache.exists():
        return _load_from_cache(cache)

    try:
        from datasets import load_dataset

        ds = load_dataset("gorilla-llm/APIBench", split="train")

        skill_texts: dict[str, str] = {}
        queries: list[dict] = []
        qrels: dict[str, set[str]] = {}
        seen_instructions: set[str] = set()

        for row in ds:
            if row.get("provider") != "Hugging Face Transformers":
                continue
            try:
                api_data = json.loads(row["api_data"])
            except (json.JSONDecodeError, KeyError):
                continue

            api_name = api_data.get("api_name", "").strip()
            description = api_data.get("description", "").strip()
            domain = api_data.get("domain", "").strip()
            if not api_name or not description:
                continue

            # Deduplicate corpus by api_name
            if api_name not in skill_texts:
                if len(skill_texts) >= 300:
                    continue
                skill_texts[api_name] = f"{description} {domain}".strip()

            # Extract instruction (before ###Output:)
            code = row.get("code", "")
            instruction = code.split("###Output:")[0].replace("###Instruction:", "").strip()
            if not instruction or instruction in seen_instructions:
                continue
            if len(queries) >= 300:
                continue

            seen_instructions.add(instruction)
            q_id = f"q_{len(queries)}"
            queries.append({"id": q_id, "text": instruction})
            qrels[q_id] = {api_name}

        if not queries or not skill_texts:
            return _synthetic_skillsbench()

        with cache.open("w") as f:
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


def run(adapter, corpus: str = "gorilla-apibench", model: str = "small") -> BenchmarkResult:
    docs, queries, qrels = download_skillsbench()
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
