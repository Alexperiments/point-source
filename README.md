# point-source [WIP]

A RAG (Retrieval-Augmented Generation) tool for querying astrophysics papers from ArXiv.

## The Stack

| Component | Tool |
| --- | --- |
| **Logic/Agents** | Pydantic AI |
| **API** | FastAPI |
| **Database** | PostgreSQL + `pgvector` |
| **ORM / Migrations** | SQLAlchemy & Alembic |
| **LLM Gateway** | LiteLLM Proxy |
| **Caching** | Redis |
| **Observability** | Logfire |

Most RAG projects rely on high-level wrappers like LangChain. This project intentionally avoids high-levels wrappers like LangChain or LlamaIndex. I wanted to maintain transparency, have a lot of flexbility and having more fun building the logic from scratch.

## Roadmap

* [x] Initial Scaffolding
* [x] Data Modeling
* [ ] ~~ArXiv PDF Parsers (Handling LaTeX/Math) (skipping for now)~~
* [x] Chunking with token-aware hierarchical splitter, math-citation-aware
* [x] Embedding benchmark (evaluation dataset + metrics)
* [x] Create embeddings
* [x] Vector Search Implementation
* [x] Hybrid Search Implementation
* [x] Generation & Attribution
* [ ] CLI
* [ ] Web UI

## TODO and possible improvements
- [ ] Allow the agent to evaluate the retrieved chunks and follow-up with a smart search to get more context from the retrieved papers (e.g. target the abstract or different sections that are cited in the retrieved chunk, to enrich the final result).
- [ ] Implement re-ranking.
- [ ] Improve agent latency.

## Setup & Usage

WIP

## Contributing

Open to PRs once the base architecture is settled.

## Acknowledgement

This project is based on the [pydantic-ai-production-ready-template](https://github.com/m7mdhka/pydantic-ai-production-ready-template).

---
