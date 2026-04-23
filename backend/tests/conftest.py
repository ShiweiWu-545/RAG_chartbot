"""
Shared fixtures and mocks for testing
"""

import os
import sys
from unittest.mock import MagicMock, Mock

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


# Define SearchResults locally to avoid importing vector_store
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

    # Create a mock tool call
    tool_call = MagicMock()
    tool_call.id = "call_123"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "Python basics"}'

    response.choices[0].finish_reason = "tool_calls"
    response.choices[0].message.content = "Let me search for that."
    response.choices[0].message.tool_calls = [tool_call]

    return response
