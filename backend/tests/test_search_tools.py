"""
Tests for Search Tools - ToolManager.execute_tool() and CourseSearchTool
File: backend/search_tools.py:138-143
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from search_tools import CourseSearchTool, ToolManager


# Define SearchResults locally
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


class MockVectorStore:
    """Mock VectorStore for testing"""

    def __init__(self):
        self.last_search_kwargs = None
        self.search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
        )

    def search(self, query, course_name=None, lesson_number=None):
        self.last_search_kwargs = {
            "query": query,
            "course_name": course_name,
            "lesson_number": lesson_number,
        }
        return self.search_results


class TestToolManager:
    """Test suite for ToolManager class"""

    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager instance"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool"""
        tool = MagicMock()
        tool.get_tool_definition.return_value = {
            "type": "function",
            "function": {"name": "test_tool", "description": "A test tool"},
        }
        tool.execute.return_value = "Tool executed successfully"
        return tool

    def test_execute_tool_found(self, tool_manager, mock_tool):
        """
        Test: When tool exists, verify execute() is called correctly
        File: backend/search_tools.py:138-143
        """
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", arg1="value1", arg2="value2")

        assert result == "Tool executed successfully"
        mock_tool.execute.assert_called_once_with(arg1="value1", arg2="value2")

    def test_execute_tool_not_found(self, tool_manager):
        """
        Test: When tool does not exist, verify error message is returned
        File: backend/search_tools.py:138-143
        """
        result = tool_manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result
        assert "nonexistent_tool" in result

    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test: Verify tool definitions are returned correctly"""
        tool_manager.register_tool(mock_tool)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["function"]["name"] == "test_tool"

    def test_get_tool_allowed_arguments(self, tool_manager, mock_tool):
        """Test: Verify allowed tool arguments are derived from the tool definition"""
        mock_tool.get_tool_definition.return_value["function"]["parameters"] = {
            "type": "object",
            "properties": {"query": {"type": "string"}, "course_name": {"type": "string"}},
        }
        tool_manager.register_tool(mock_tool)

        allowed_args = tool_manager.get_tool_allowed_arguments("test_tool")

        assert allowed_args == {"query", "course_name"}

    def test_get_last_sources(self, tool_manager, mock_tool):
        """Test: Verify sources from last search are retrieved"""
        # Create a mock tool that tracks last_sources
        mock_tool.last_sources = ["Course A - Lesson 1", "Course B - Lesson 2"]
        tool_manager.register_tool(mock_tool)

        sources = tool_manager.get_last_sources()

        assert sources == ["Course A - Lesson 1", "Course B - Lesson 2"]

    def test_reset_sources(self, tool_manager, mock_tool):
        """Test: Verify sources are reset after retrieval"""
        mock_tool.last_sources = ["Course A - Lesson 1"]
        tool_manager.register_tool(mock_tool)

        tool_manager.reset_sources()

        assert mock_tool.last_sources == []

    def test_execute_tool_returns_search_output(self, tool_manager):
        """
        Test: Verify ToolManager.execute_tool() returns the formatted output from CourseSearchTool.
        """
        mock_vector_store = MockVectorStore()
        course_search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(course_search_tool)

        result = tool_manager.execute_tool("search_course_content", query="Python")

        assert "[Test - Lesson 1]" in result
        assert "Test content" in result

    def test_execute_tool_returns_controlled_error_on_exception(self, tool_manager, mock_tool):
        """Test: Verify execution exceptions are returned as controlled errors."""
        mock_tool.execute.side_effect = RuntimeError("backend exploded")
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", query="Python")

        assert result == "Tool execution failed: backend exploded"


class TestCourseSearchTool:
    """Test suite for CourseSearchTool class"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock VectorStore"""
        return MockVectorStore()

    @pytest.fixture
    def course_search_tool(self, mock_vector_store):
        """Create a CourseSearchTool with mock store"""
        return CourseSearchTool(mock_vector_store)

    def test_execute_with_results(self, course_search_tool, mock_vector_store):
        """
        Test: When search returns results, verify formatting and source tracking
        File: backend/search_tools.py:55-89
        """
        mock_results = SearchResults(
            documents=["Python is a programming language.", "It supports multiple paradigms."],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1, "chunk_index": 0},
                {"course_title": "Python Basics", "lesson_number": 2, "chunk_index": 0},
            ],
            distances=[0.1, 0.2],
        )
        mock_vector_store.search_results = mock_results

        result = course_search_tool.execute(query="Python")

        assert "Python Basics" in result
        assert "Lesson 1" in result
        assert "programming language" in result

    def test_execute_with_empty_results(self, course_search_tool, mock_vector_store):
        """
        Test: When search returns empty results, verify proper message
        File: backend/search_tools.py:55-89
        """
        mock_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search_results = mock_results

        result = course_search_tool.execute(query="Nonexistent topic")

        assert "No relevant content found" in result

    def test_execute_with_error(self, course_search_tool, mock_vector_store):
        """
        Test: When search returns error, verify error message is returned
        File: backend/search_tools.py:55-89
        """
        mock_results = SearchResults.empty("Search error: Invalid query")
        mock_vector_store.search_results = mock_results

        result = course_search_tool.execute(query="test")

        assert "Search error" in result

    def test_execute_with_course_name_filter(self, course_search_tool, mock_vector_store):
        """
        Test: When course_name is provided, verify it's passed to search
        File: backend/search_tools.py:55-89
        """
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search_results = mock_results

        course_search_tool.execute(query="content", course_name="Test Course")

        assert mock_vector_store.last_search_kwargs == {
            "query": "content",
            "course_name": "Test Course",
            "lesson_number": None,
        }

    def test_execute_with_lesson_number_filter(self, course_search_tool, mock_vector_store):
        """
        Test: When lesson_number is provided, verify it's passed to search
        File: backend/search_tools.py:55-89
        """
        mock_results = SearchResults(
            documents=["Content"],
            metadata=[{"course_title": "Test", "lesson_number": 3}],
            distances=[0.1],
        )
        mock_vector_store.search_results = mock_results

        course_search_tool.execute(query="content", lesson_number=3)

        assert mock_vector_store.last_search_kwargs == {
            "query": "content",
            "course_name": None,
            "lesson_number": 3,
        }

    def test_sources_are_tracked(self, course_search_tool, mock_vector_store):
        """
        Test: Verify sources from search results are tracked for retrieval
        File: backend/search_tools.py:55-89
        """
        mock_results = SearchResults(
            documents=["Content A"],
            metadata=[{"course_title": "Course A", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search_results = mock_results

        course_search_tool.execute(query="test")

        assert course_search_tool.last_sources == ["Course A - Lesson 1"]

    def test_get_tool_definition(self, course_search_tool):
        """
        Test: Verify tool definition format for OpenAI compatibility
        File: backend/search_tools.py:27-53
        """
        definition = course_search_tool.get_tool_definition()

        assert definition["type"] == "function"
        assert definition["function"]["name"] == "search_course_content"
        assert "query" in definition["function"]["parameters"]["properties"]
        assert "course_name" in definition["function"]["parameters"]["properties"]
        assert "lesson_number" in definition["function"]["parameters"]["properties"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
