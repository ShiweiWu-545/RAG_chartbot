"""
Tests for RAG System query/search orchestration without touching real ChromaDB or model downloads.
File: backend/rag_system.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from rag_system import RAGSystem


class TestRAGSystem:
    """Test suite for RAGSystem query flow"""

    @pytest.fixture
    def mock_config(self):
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
    def rag_and_deps(self, mock_config):
        with (
            patch("rag_system.DocumentProcessor") as mock_doc_processor_cls,
            patch("rag_system.VectorStore") as mock_vector_store_cls,
            patch("rag_system.AIGenerator") as mock_ai_generator_cls,
            patch("rag_system.SessionManager") as mock_session_manager_cls,
            patch("rag_system.ToolManager") as mock_tool_manager_cls,
            patch("rag_system.CourseSearchTool") as mock_search_tool_cls,
        ):
            mock_doc_processor = MagicMock()
            mock_vector_store = MagicMock()
            mock_ai_generator = MagicMock()
            mock_session_manager = MagicMock()
            mock_tool_manager = MagicMock()
            mock_search_tool = MagicMock()

            mock_ai_generator.generate_response.return_value = (
                "This is a test response about course materials."
            )
            mock_session_manager.get_conversation_history.return_value = None
            mock_tool_manager.get_tool_definitions.return_value = [
                {"type": "function", "function": {"name": "search_course_content"}}
            ]
            mock_tool_manager.get_last_sources.return_value = ["Python Basics - Lesson 1"]

            mock_doc_processor_cls.return_value = mock_doc_processor
            mock_vector_store_cls.return_value = mock_vector_store
            mock_ai_generator_cls.return_value = mock_ai_generator
            mock_session_manager_cls.return_value = mock_session_manager
            mock_tool_manager_cls.return_value = mock_tool_manager
            mock_search_tool_cls.return_value = mock_search_tool

            rag = RAGSystem(mock_config)

            yield {
                "rag": rag,
                "document_processor": mock_doc_processor,
                "vector_store": mock_vector_store,
                "ai_generator": mock_ai_generator,
                "session_manager": mock_session_manager,
                "tool_manager": mock_tool_manager,
                "search_tool": mock_search_tool,
            }

    def test_init_registers_search_tool(self, rag_and_deps):
        deps = rag_and_deps

        deps["tool_manager"].register_tool.assert_called_once_with(deps["search_tool"])

    def test_query_without_session_does_not_create_or_store_session(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]

        response, sources = rag.query("What is Python?")

        deps["session_manager"].get_conversation_history.assert_not_called()
        deps["session_manager"].add_exchange.assert_not_called()
        assert response == "This is a test response about course materials."
        assert sources == ["Python Basics - Lesson 1"]

    def test_query_with_session_uses_existing_history(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]

        rag.query("What is Python?", session_id="existing_session")

        deps["session_manager"].get_conversation_history.assert_called_once_with("existing_session")

    def test_query_calls_ai_generator_with_tools_and_manager(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]

        rag.query("What is Python?")

        deps["ai_generator"].generate_response.assert_called_once()
        call_kwargs = deps["ai_generator"].generate_response.call_args.kwargs
        assert call_kwargs["tools"] == deps["tool_manager"].get_tool_definitions.return_value
        assert call_kwargs["tool_manager"] is deps["tool_manager"]

    def test_query_retrieves_and_resets_sources(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]

        _, sources = rag.query("What is Python?")

        deps["tool_manager"].get_last_sources.assert_called_once()
        deps["tool_manager"].reset_sources.assert_called_once()
        assert sources == ["Python Basics - Lesson 1"]

    def test_query_updates_conversation_history_only_with_session(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]

        rag.query("What is Python?", session_id="session_1")

        deps["session_manager"].add_exchange.assert_called_once()
        call_args = deps["session_manager"].add_exchange.call_args.args
        assert call_args[0] == "session_1"
        assert call_args[1] == "What is Python?"
        assert call_args[2] == "This is a test response about course materials."

    def test_query_builds_prompt_correctly(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]

        rag.query("What is Python?")

        call_kwargs = deps["ai_generator"].generate_response.call_args.kwargs
        assert "Answer this question about course materials" in call_kwargs["query"]
        assert "What is Python?" in call_kwargs["query"]

    def test_query_passes_conversation_history_through(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]
        deps["session_manager"].get_conversation_history.return_value = (
            "User: What is Python?\nAssistant: Python is a language."
        )

        rag.query("Tell me more", session_id="session_1")

        call_kwargs = deps["ai_generator"].generate_response.call_args.kwargs
        assert call_kwargs["conversation_history"] == (
            "User: What is Python?\nAssistant: Python is a language."
        )

    def test_query_propagates_ai_errors(self, rag_and_deps):
        deps = rag_and_deps
        rag = deps["rag"]
        deps["ai_generator"].generate_response.side_effect = RuntimeError("tool execution failed")

        with pytest.raises(RuntimeError) as exc_info:
            rag.query("Summarize lesson 1", session_id="session_1")

        assert "tool execution failed" in str(exc_info.value)
        deps["tool_manager"].get_last_sources.assert_not_called()
