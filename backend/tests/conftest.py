"""
Shared fixtures and mocks for testing
"""

import os
import sys
from unittest.mock import MagicMock, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


class SearchResults:
    """Mock SearchResults class for testing"""

    def __init__(self, documents=None, metadata=None, distances=None, error=None):
        self.documents = documents or []
        self.metadata = metadata or []
        self.distances = distances or []
        self.error = error

    @classmethod
    def from_chroma(cls, chroma_results):
        return cls(
            documents=chroma_results.get("documents", [[]])[0],
            metadata=chroma_results.get("metadatas", [[]])[0],
            distances=chroma_results.get("distances", [[]])[0],
        )

    @classmethod
    def empty(cls, error_msg):
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self):
        return len(self.documents) == 0


@pytest.fixture
def mock_search_results():
    """Create a mock SearchResults object with sample data"""
    return SearchResults(
        documents=[
            "This is sample course content about Python programming.",
            "This is more content about machine learning basics.",
        ],
        metadata=[
            {"course_title": "Python Basics", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "ML Introduction", "lesson_number": 2, "chunk_index": 0},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_vector_store(mock_search_results):
    """Create a mock VectorStore with pre-configured search results"""
    store = MagicMock()
    store.search.return_value = mock_search_results
    return store


@pytest.fixture
def tool_manager_with_mock_store(mock_vector_store):
    """Create a ToolManager with a mocked CourseSearchTool"""
    from search_tools import CourseSearchTool, ToolManager

    tm = ToolManager()
    search_tool = CourseSearchTool(mock_vector_store)
    tm.register_tool(search_tool)
    return tm


@pytest.fixture
def mock_ai_response():
    """Create a mock OpenAI completion response without tool calls"""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "This is a test response from the AI."
    return response


@pytest.fixture
def mock_ai_response_with_tool_calls():
    """Create a mock OpenAI completion response with tool calls"""
    response = MagicMock()
    response.choices = [MagicMock()]

    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "Python basics"}'

    response.choices[0].finish_reason = "tool_calls"
    response.choices[0].message.content = "Let me search for that."
    response.choices[0].message.tool_calls = [tool_call]

    return response


@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    config = MagicMock()
    config.MINIMAX_API_KEY = "test_key"
    config.MINIMAX_MODEL = "test-model"
    config.MINIMAX_BASE_URL = "https://test.api"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def mock_rag_system(mock_config):
    """Create a mock RAGSystem with pre-configured responses"""
    from unittest.mock import patch

    with patch("rag_system.DocumentProcessor"), \
         patch("rag_system.VectorStore") as mock_vs_cls, \
         patch("rag_system.SessionManager") as mock_sm_cls, \
         patch("rag_system.AIGenerator") as mock_ai_cls, \
         patch("rag_system.ToolManager") as mock_tm_cls, \
         patch("rag_system.CourseSearchTool"):

        mock_vs = MagicMock()
        mock_sm = MagicMock()
        mock_ai = MagicMock()
        mock_tm = MagicMock()

        mock_sm.create_session.return_value = "test_session_123"
        mock_sm.get_conversation_history.return_value = None
        mock_ai.generate_response.return_value = "Test AI response about course materials."
        mock_tm.get_tool_definitions.return_value = [
            {"type": "function", "function": {"name": "search_course_content"}}
        ]
        mock_tm.get_last_sources.return_value = ["Python Basics - Lesson 1"]

        mock_vs_cls.return_value = mock_vs
        mock_sm_cls.return_value = mock_sm
        mock_ai_cls.return_value = mock_ai
        mock_tm_cls.return_value = mock_tm

        from rag_system import RAGSystem
        rag = RAGSystem(mock_config)

        yield {
            "rag": rag,
            "mock_vs": mock_vs,
            "mock_sm": mock_sm,
            "mock_ai": mock_ai,
            "mock_tm": mock_tm
        }


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app with mocked dependencies"""
    from unittest.mock import patch
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    mock_rag = mock_rag_system["rag"]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()

            answer, sources = mock_rag.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Query failed. Please try again.")

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to load course stats.")

    return app


@pytest.fixture
def test_client(test_app):
    """Create a TestClient for the test FastAPI app"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)