# Repository Guidelines

## Project Structure & Module Organization
This repository is a small full-stack RAG chatbot. `backend/` contains the FastAPI service, retrieval pipeline, and tests. Key backend modules include `app.py` for API routes and static file mounting, `rag_system.py` for orchestration, `search_tools.py` for tool execution, and `vector_store.py` for ChromaDB access. `backend/tests/` holds `pytest` suites for the RAG system, tool layer, and AI generator. `frontend/` contains the static UI (`index.html`, `script.js`, `style.css`). Source course files live in `docs/`, and persistent local vector data is stored in `backend/chroma_db/`.

## Build, Test, and Development Commands
Always use `uv` for dependency installation and command execution. In PowerShell, set the local `uv` path first with `$env:PATH += ";C:\Users\86787\.local\bin"`. Install dependencies from the repository root with `uv sync`. Run the backend from `backend/` with `uv run uvicorn app:app --reload --port 8000`. If you want the default startup behavior, keep `docs/` populated so the FastAPI startup hook can ingest files automatically. Run tests from `backend/` with `uv run pytest tests`. For a narrower loop, use `uv run pytest tests/test_rag_system.py` or similar per module.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, type hints on public methods, and concise docstrings for non-obvious behavior. Use `snake_case` for functions, methods, and module names; use `PascalCase` for classes such as `RAGSystem` and `CourseSearchTool`; keep configuration constants uppercase in `config.py`. Keep modules focused: API concerns in `app.py`, orchestration in `rag_system.py`, storage logic in `vector_store.py`.

## Testing Guidelines
Tests use `pytest` with fixtures in `backend/tests/conftest.py` and `unittest.mock` for API and vector store isolation. Name new tests `test_<behavior>.py` and group related assertions in `Test...` classes when a file covers one component. Add or update tests for any change to query flow, tool execution, or source formatting. Prefer deterministic mocks over live API calls.

## Commit & Pull Request Guidelines
The current Git history is minimal (`Initial commit: Course Materials RAG System with MiniMax`), so use short, imperative commit subjects, for example `Add source link formatting to search results`. Keep one logical change per commit. Pull requests should include a brief summary, affected areas (`backend`, `frontend`, `docs`), test evidence (`uv run pytest tests`), and screenshots when UI behavior changes.

## Security & Configuration Tips
Keep secrets in `.env`; do not commit API keys. Use `.env.example` as the template for local setup. Review changes touching `backend/ai_generator.py` carefully, especially tool-call argument handling and external API configuration. Avoid committing generated ChromaDB data unless intentionally updating local fixtures.

## Agent-Specific Notes
When working in this repository, prefer `uv` over direct `python`, `pip`, or other runners for install, test, and local execution steps. In PowerShell sessions where `uv` is missing from `PATH`, run `$env:PATH += ";C:\Users\86787\.local\bin"` before any `uv` command.
