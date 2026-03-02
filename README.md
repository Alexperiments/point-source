# Point-source

Point-source is an astrophysics RAG assistant.
It lets you chat with papers and get grounded, citation-oriented answers from a curated scientific corpus.

> [!NOTE]
>
> About this version:
> Point-source is currently in alpha. The papers available to the system are a curated subset of arXiv rather than the full corpus: specifically, papers in the `astro-ph` category from the ar5iv HTML project, filtered to documents without parsing errors to prioritize answer quality and citation reliability. The current source dataset is `marin-community/ar5iv-no-problem-markdown`.
>
> Implications of this limited sample:
> - Coverage is not up to date (current corpus snapshot has an approximate cutoff around 2024).
> - Coverage is not exhaustive.
> - Some relevant papers are excluded because they fail parsing/quality filters.


## Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS
- Backend: FastAPI, Pydantic AI, LiteLLM
- Data: PostgreSQL + pgvector, Redis

## Development (Quick Start)

### Prerequisites

- Docker + Docker Compose
- Node.js 20+
- Python 3.12 + `uv`

### 1. Configure environment

- Backend env file is `backend/.env.development`
- Frontend env file:

```bash
cd frontend
cp .env.example .env
```

If you run the frontend with Vite (`npm run dev`), set:

```bash
VITE_API_BASE_URL="http://localhost:8000"
```

### 2. Build frontend assets (for backend-served UI)

```bash
cd frontend
npm ci
npm run build
```

### 3. Start the development stack

From repo root:

```bash
make docker-dev-up
```

Open:

- App: `http://127.0.0.1:8000`


## Contributing

Contributions are welcome.

1. Open an issue (bug report, question, or feature request)
2. Fork and create a branch from `main`
3. Keep changes focused and include tests when relevant
4. Open a PR with a clear description and rationale

Small improvements, docs fixes, and UX polish are all appreciated.

## Acknowledgement

The original backend structure of this project is based on the [pydantic-ai-production-ready-template](https://github.com/m7mdhka/pydantic-ai-production-ready-template).
