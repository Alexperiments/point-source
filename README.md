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

## Repository Layout

- `backend/`: FastAPI app, tests, migrations, scripts, and Python project files.
- `infra/`: Docker Compose, LiteLLM, Grafana, and Prometheus configuration.
- `frontend/`: React + Vite frontend application.
- `deployment/`: Reserved for deployment manifests and automation.

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
- [ ] Modify the retrieval to correctly provide papers with metadata.
- [ ] In the new Arxiv ingestion there are more than 80k astro-ph papers. They need chunking and embedding.

## Setup & Usage

### Local Development

1. Start infra services:
   - `make docker-dev-up`
2. Start backend:
   - `make run-dev`
3. Start frontend dev server:
   - `cd frontend && npm ci && npm run dev`

### Single-App Mode (backend serves frontend)

1. Build frontend assets:
   - `cd frontend && npm ci && npm run build`
2. Start backend:
   - `make run-dev`
3. Open `http://localhost:8000`.

## Contributing

Open to PRs once the base architecture is settled.

## Acknowledgement

This project is based on the [pydantic-ai-production-ready-template](https://github.com/m7mdhka/pydantic-ai-production-ready-template).

---
