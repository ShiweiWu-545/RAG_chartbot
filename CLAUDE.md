# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Course Materials RAG System - a web application that answers questions about course materials using semantic search (ChromaDB) and Claude AI (Anthropic). Users query course documents through a web interface and receive AI-generated answers with sources.

## Running the Application

```bash
# Install dependencies
uv sync

# Set API key (create .env in project root)
echo "ANTHROPIC_API_KEY=sk_ant_..." > .env

# Run application
./run.sh
# Or manually: cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web interface: http://localhost:8000
- API docs: http://localhost:8000/docs

## Architecture

```
frontend/          - Vanilla HTML/CSS/JS web interface
backend/
  app.py           - FastAPI entry point, serves API + frontend
  rag_system.py    - RAG orchestration (coordinates all components)
  vector_store.py  - ChromaDB wrapper (two collections: course_catalog, course_content)
  document_processor.py - Parses course documents into chunks
  ai_generator.py  - Claude API integration with tool execution
  search_tools.py  - ToolManager + CourseSearchTool for semantic search
  session_manager.py - Conversation history per session
  config.py        - Configuration (embedding model, chunk size, etc.)
  models.py        - Pydantic dataclasses (Course, Lesson, CourseChunk)
docs/              - Course materials (source documents for the knowledge base)
```

## Query Flow

1. Frontend sends POST `/api/query` with `{query, session_id}`
2. `RAGSystem.query()` coordinates:
   - `SessionManager` retrieves conversation history
   - `AIGenerator.generate_response()` calls Claude with tools
3. If Claude decides to search (stop_reason == "tool_use"):
   - `ToolManager.execute_tool("search_course_content", ...)`
   - `VectorStore.search()` queries ChromaDB (course_content collection)
   - Results fed back to Claude for final answer
4. Response and sources returned to frontend

## Key Implementation Details

- **Two Claude API calls per query**: First call determines if search needed, second call generates answer with search results
- **Tool-based RAG**: Claude autonomously decides when to invoke `search_course_content` tool
- **Embedding**: sentence-transformers `all-MiniLM-L6-v2`, chunk size 800 chars, overlap 100
- **ChromaDB collections**: `course_catalog` (course metadata) and `course_content` (chunked content)
- **Session history**: `MAX_HISTORY=2` (keeps last 2 exchanges)

## Configuration

All settings in `backend/config.py` via dataclass:
- `MINIMAX_MODEL`: MiniMax-M2.7
- `MINIMAX_BASE_URL`: https://api.minimax.chat/v1
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2
- `CHUNK_SIZE`: 800, `CHUNK_OVERLAP`: 100
- `MAX_RESULTS`: 5, `MAX_HISTORY`: 2
- `CHROMA_PATH`: ./chroma_db

## Development

- **Always use `uv run python <file>`** instead of `python <file>`
- **Always use `uv add <package>`** instead of `pip install <package>`

## Tech Stack

- Python 3.13+, FastAPI, Uvicorn
- ChromaDB (PersistentClient, SentenceTransformerEmbeddingFunction)
- Anthropic SDK (anthropic Python package)
- sentence-transformers for embeddings
- uv for package management
