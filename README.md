# point-source

A RAG (Retrieval-Augmented Generation) tool for querying astrophysics papers from ArXiv.

**Current Status:** Early development. Focused on data modeling and vector schema.

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
* [x] Data Modeling (Current focus)
* [ ] ArXiv PDF Parsers (Handling LaTeX/Math) (skipping for now)
* [ ] Chunking & Embedding logic
* [ ] Vector Search Implementation
* [ ] Generation & Attribution
* [ ] CLI & Web UI

## Setup & Usage

WIP

## Contributing

Open to PRs once the base architecture is settled.

## Acknowledgement

This project is based on the [pydantic-ai-production-ready-template](https://github.com/m7mdhka/pydantic-ai-production-ready-template).

---
