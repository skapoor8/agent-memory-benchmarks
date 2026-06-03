"""
Hardcoded exemplar queries for the Jupyter Book notebook "Example Queries" sections.
Writes scripts/sample_queries.json — used by notebooks without needing a live index.
Run via: uv run python scripts/sample_queries.py
"""

import json
import pathlib

EXEMPLARS = {
    "code-finding": [
        {
            "query": "Fix the TypeError when None is passed to the tokenizer encode method",
            "results": [
                {
                    "id": "src/transformers/tokenization_utils.py",
                    "score": 0.84,
                    "snippet": "def encode(self, text, add_special_tokens=True, **kwargs):",
                },
                {
                    "id": "src/transformers/tokenization_utils_base.py",
                    "score": 0.79,
                    "snippet": "def _encode_plus(self, text, ...) -> BatchEncoding:",
                },
                {
                    "id": "tests/test_tokenization_common.py",
                    "score": 0.61,
                    "snippet": "def test_encode_decode_with_spaces(self):",
                },
                {
                    "id": "src/transformers/models/bert/tokenization_bert.py",
                    "score": 0.58,
                    "snippet": "class BertTokenizer(PreTrainedTokenizer):",
                },
                {
                    "id": "src/transformers/utils/generic.py",
                    "score": 0.42,
                    "snippet": "def is_tensor(x): return isinstance(x, torch.Tensor)",
                },
            ],
        },
        {
            "query": "AttributeError: 'NoneType' object has no attribute 'shape' in attention mask computation",
            "results": [
                {
                    "id": "src/transformers/modeling_utils.py",
                    "score": 0.81,
                    "snippet": "def get_extended_attention_mask(self, attention_mask, input_shape):",
                },
                {
                    "id": "src/transformers/models/bert/modeling_bert.py",
                    "score": 0.76,
                    "snippet": "class BertSelfAttention(nn.Module):",
                },
                {
                    "id": "src/transformers/models/gpt2/modeling_gpt2.py",
                    "score": 0.63,
                    "snippet": "def _attn(self, query, key, value, attention_mask=None, head_mask=None):",
                },
                {
                    "id": "src/transformers/generation/utils.py",
                    "score": 0.55,
                    "snippet": "def _prepare_attention_mask_for_generation(self, inputs, ...):",
                },
                {"id": "tests/test_modeling_common.py", "score": 0.38, "snippet": "def test_attention_outputs(self):"},
            ],
        },
        {
            "query": "IndexError in DataCollatorForSeq2Seq when padding batch to max length",
            "results": [
                {
                    "id": "src/transformers/data/data_collator.py",
                    "score": 0.88,
                    "snippet": "class DataCollatorForSeq2Seq:",
                },
                {
                    "id": "src/transformers/trainer.py",
                    "score": 0.67,
                    "snippet": "def _get_collator_with_removed_columns(self, data_collator, ...):",
                },
                {
                    "id": "src/transformers/tokenization_utils_base.py",
                    "score": 0.59,
                    "snippet": "def pad(self, encoded_inputs, padding=True, ...):",
                },
                {
                    "id": "examples/pytorch/translation/run_translation.py",
                    "score": 0.44,
                    "snippet": "data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, ...)",
                },
                {
                    "id": "tests/test_data_collator.py",
                    "score": 0.41,
                    "snippet": "def test_data_collator_for_seq2seq(self):",
                },
            ],
        },
    ],
    "doc-search": [
        {
            "query": "How do I add OAuth2 with password flow to a FastAPI application?",
            "results": [
                {
                    "id": "docs/tutorial/security/oauth2-jwt.md",
                    "score": 0.91,
                    "snippet": "# OAuth2 with Password (and hashing), Bearer with JWT tokens",
                },
                {
                    "id": "docs/tutorial/security/simple-oauth2.md",
                    "score": 0.87,
                    "snippet": "# Simple OAuth2 with Password and Bearer",
                },
                {"id": "docs/tutorial/security/oauth2-scopes.md", "score": 0.74, "snippet": "# OAuth2 scopes"},
                {"id": "docs/tutorial/security/http-basic-auth.md", "score": 0.52, "snippet": "# HTTP Basic Auth"},
                {
                    "id": "docs/tutorial/middleware.md",
                    "score": 0.31,
                    "snippet": "# Middleware — adding custom middleware to your app",
                },
            ],
        },
        {
            "query": "How do I return a custom HTTP status code from a path operation?",
            "results": [
                {
                    "id": "docs/tutorial/response-status-code.md",
                    "score": 0.93,
                    "snippet": "# Response Status Code — use the `status_code` parameter",
                },
                {
                    "id": "docs/tutorial/handling-errors.md",
                    "score": 0.78,
                    "snippet": "# Handling Errors — raise HTTPException with status_code",
                },
                {
                    "id": "docs/tutorial/path-operation-configuration.md",
                    "score": 0.65,
                    "snippet": "# Path Operation Configuration — status_code, tags, summary",
                },
                {
                    "id": "docs/tutorial/additional-responses.md",
                    "score": 0.58,
                    "snippet": "# Additional Responses in OpenAPI",
                },
                {
                    "id": "docs/advanced/response-directly.md",
                    "score": 0.44,
                    "snippet": "# Return a Response Directly using JSONResponse",
                },
            ],
        },
        {
            "query": "What is the difference between async def and def in FastAPI route handlers?",
            "results": [
                {
                    "id": "docs/async.md",
                    "score": 0.95,
                    "snippet": "# Concurrency and async / await — when to use async def vs def",
                },
                {
                    "id": "docs/tutorial/first-steps.md",
                    "score": 0.61,
                    "snippet": "# First Steps — define your first path operation with async def",
                },
                {"id": "docs/tutorial/path-params.md", "score": 0.48, "snippet": "# Path Parameters"},
                {
                    "id": "docs/deployment/server-workers.md",
                    "score": 0.43,
                    "snippet": "# Server Workers — Gunicorn with Uvicorn Workers",
                },
                {"id": "docs/tutorial/background-tasks.md", "score": 0.39, "snippet": "# Background Tasks"},
            ],
        },
    ],
    "episodic-memory": [
        {
            "query": "Why did we choose Paxos-based consensus instead of Raft for the storage layer?",
            "results": [
                {
                    "id": "docs/adrs/2023-08-storage-consensus.md",
                    "score": 0.89,
                    "snippet": "# ADR: Storage Consensus Protocol — Decision: Use multi-Paxos for WAL replication",
                },
                {
                    "id": "docs/adrs/2023-06-replication-model.md",
                    "score": 0.74,
                    "snippet": "# ADR: Replication Model — evaluated Raft, Paxos, and Viewstamped Replication",
                },
                {
                    "id": "docs/adrs/2023-09-safekeeper-protocol.md",
                    "score": 0.68,
                    "snippet": "# ADR: Safekeeper WAL Protocol — quorum writes across 3 safekeepers",
                },
                {
                    "id": "docs/adrs/2022-11-architecture-overview.md",
                    "score": 0.51,
                    "snippet": "# Architecture Overview — separation of storage and compute",
                },
                {
                    "id": "docs/adrs/2024-01-durability-guarantees.md",
                    "score": 0.38,
                    "snippet": "# ADR: Durability Guarantees — RPO and RTO targets",
                },
            ],
        },
        {
            "query": "What drove the decision to use a 8kb page format rather than a variable-size format?",
            "results": [
                {
                    "id": "docs/adrs/2022-12-page-format.md",
                    "score": 0.92,
                    "snippet": "# ADR: Page Format — Decision: fixed 8kb pages for PostgreSQL compatibility",
                },
                {
                    "id": "docs/adrs/2023-02-buffer-cache.md",
                    "score": 0.71,
                    "snippet": "# ADR: Buffer Cache Design — page size alignment with OS huge pages",
                },
                {
                    "id": "docs/adrs/2023-05-storage-format.md",
                    "score": 0.63,
                    "snippet": "# ADR: Storage Format — columnar vs row-oriented consideration",
                },
                {
                    "id": "docs/adrs/2022-11-architecture-overview.md",
                    "score": 0.49,
                    "snippet": "# Architecture Overview — storage node design",
                },
                {
                    "id": "docs/adrs/2023-10-compaction.md",
                    "score": 0.35,
                    "snippet": "# ADR: Compaction Strategy — tiered compaction for L0 files",
                },
            ],
        },
        {
            "query": "When did we decide to use Kubernetes for the control plane and why?",
            "results": [
                {
                    "id": "docs/adrs/2023-03-control-plane.md",
                    "score": 0.88,
                    "snippet": "# ADR: Control Plane Infrastructure — Decision: Kubernetes with custom operators",
                },
                {
                    "id": "docs/adrs/2023-07-scaling-model.md",
                    "score": 0.69,
                    "snippet": "# ADR: Scaling Model — horizontal pod autoscaling for compute nodes",
                },
                {
                    "id": "docs/adrs/2022-11-architecture-overview.md",
                    "score": 0.57,
                    "snippet": "# Architecture Overview — control plane responsibilities",
                },
                {"id": "docs/adrs/2023-11-multi-region.md", "score": 0.44, "snippet": "# ADR: Multi-Region Deployment"},
                {
                    "id": "docs/adrs/2024-02-observability.md",
                    "score": 0.31,
                    "snippet": "# ADR: Observability Stack — Prometheus, Grafana, OpenTelemetry",
                },
            ],
        },
    ],
    "skill-search": [
        {
            "query": "I need to classify the sentiment of customer product reviews",
            "results": [
                {
                    "id": "cardiffnlp/twitter-roberta-base-sentiment",
                    "score": 0.87,
                    "snippet": "Sentiment analysis — classifies text as positive, neutral, or negative",
                },
                {
                    "id": "distilbert-base-uncased-finetuned-sst-2-english",
                    "score": 0.83,
                    "snippet": "DistilBERT fine-tuned on SST-2 for binary sentiment classification",
                },
                {
                    "id": "nlptown/bert-base-multilingual-uncased-sentiment",
                    "score": 0.76,
                    "snippet": "Multilingual sentiment analysis — predicts 1 to 5 star ratings",
                },
                {
                    "id": "siebert/sentiment-roberta-large-english",
                    "score": 0.71,
                    "snippet": "RoBERTa-large fine-tuned for sentiment analysis on English text",
                },
                {
                    "id": "finiteautomata/bertweet-base-sentiment-analysis",
                    "score": 0.64,
                    "snippet": "BERTweet fine-tuned on English tweets for sentiment classification",
                },
            ],
        },
        {
            "query": "Generate a caption describing what is in this image",
            "results": [
                {
                    "id": "Salesforce/blip-image-captioning-large",
                    "score": 0.91,
                    "snippet": "Image captioning — generates a natural language description of an image",
                },
                {
                    "id": "nlpconnect/vit-gpt2-image-captioning",
                    "score": 0.85,
                    "snippet": "ViT-GPT2 for image-to-text captioning",
                },
                {
                    "id": "microsoft/git-base-coco",
                    "score": 0.79,
                    "snippet": "GIT model fine-tuned on COCO for image captioning",
                },
                {
                    "id": "Salesforce/blip-image-captioning-base",
                    "score": 0.74,
                    "snippet": "BLIP base model for image captioning and VQA",
                },
                {
                    "id": "ydshieh/vit-gpt2-coco-en",
                    "score": 0.61,
                    "snippet": "ViT encoder + GPT-2 decoder trained on COCO captions",
                },
            ],
        },
        {
            "query": "Translate a document from German to English",
            "results": [
                {
                    "id": "Helsinki-NLP/opus-mt-de-en",
                    "score": 0.94,
                    "snippet": "Machine translation — German to English using MarianMT",
                },
                {
                    "id": "Helsinki-NLP/opus-mt-mul-en",
                    "score": 0.81,
                    "snippet": "Multilingual to English translation — supports 80+ source languages",
                },
                {
                    "id": "facebook/nllb-200-distilled-600M",
                    "score": 0.76,
                    "snippet": "No Language Left Behind — 200-language translation model",
                },
                {
                    "id": "Helsinki-NLP/opus-tatoeba-de-en",
                    "score": 0.72,
                    "snippet": "German-English translation trained on Tatoeba corpus",
                },
                {
                    "id": "t5-base",
                    "score": 0.55,
                    "snippet": "T5 text-to-text model — supports translation as a text task",
                },
            ],
        },
    ],
}

if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "sample_queries.json"
    out.write_text(json.dumps(EXEMPLARS, indent=2))
    print(f"wrote {out}")
