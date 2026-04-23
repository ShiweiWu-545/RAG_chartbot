"""
Integration tests for the query flow across AIGenerator, RAGSystem, and the API layer.
"""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from ai_generator import AIGenerator
from rag_system import RAGSystem


class SearchResults:
    def __init__(self, documents=None, metadata=None, distances=None, error=None):
        self.documents = documents or []
        self.metadata = metadata or []
        self.distances = distances or []
        self.error = error

    @classmethod
    def empty(cls, error_msg):
        return cls(documents=[], metadata=[], distances=[], error=error_msg)

    def is_empty(self):
        return len(self.documents) == 0


def make_tool_call(call_id, query_text):
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = f'{{"query": "{query_text}"}}'
    return tool_call


def make_response(content, tool_calls=None):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    if tool_calls is not None:
        response.choices[0].message.tool_calls = tool_calls
        response.choices[0].finish_reason = "tool_calls"
    else:
        response.choices[0].finish_reason = "stop"
    return response


class TestQueryFlowIntegration:
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

    def test_rag_system_multi_round_query_supports_follow_up_search(self, mock_config):
        with patch("ai_generator.OpenAI") as mock_openai, \
             patch("rag_system.DocumentProcessor"), \
             patch("rag_system.VectorStore") as mock_vector_store_cls, \
             patch("rag_system.SessionManager") as mock_session_manager_cls, \
             patch("rag_system.AIGenerator") as mock_ai_generator_cls:

            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            real_ai = AIGenerator(
                api_key="test_key",
                model="test-model",
                base_url="https://test.api"
            )
            real_ai.client = mock_client
            mock_ai_generator_cls.return_value = real_ai

            mock_vector_store = MagicMock()

            def search_side_effect(query, course_name=None, lesson_number=None):
                if query == "course outline for course X":
                    return SearchResults(
                        documents=["Lesson 5 title: MCP workflows"],
                        metadata=[{"course_title": "Course X", "lesson_number": 5}],
                        distances=[0.1]
                    )
                if query == "MCP workflows":
                    return SearchResults(
                        documents=["Course Y explains MCP workflows in depth."],
                        metadata=[{"course_title": "Course Y", "lesson_number": 1}],
                        distances=[0.2]
                    )
                return SearchResults.empty("No results")

            mock_vector_store.search.side_effect = search_side_effect
            mock_vector_store_cls.return_value = mock_vector_store

            mock_session_manager = MagicMock()
            mock_session_manager.get_conversation_history.return_value = None
            mock_session_manager_cls.return_value = mock_session_manager

            rag = RAGSystem(mock_config)

            mock_client.chat.completions.create.side_effect = [
                make_response(
                    "I will check the lesson title first.",
                    [make_tool_call("call_1", "course outline for course X")]
                ),
                make_response(
                    "I found the lesson title, now I will search for a matching course.",
                    [make_tool_call("call_2", "MCP workflows")]
                ),
                make_response("Course Y explains the same topic as lesson 5 of course X.")
            ]

            answer, sources = rag.query(
                "Is there any other course topic with the same theme as MCP course lesson 5?",
                session_id="session_1"
            )

            assert answer == "Course Y explains the same topic as lesson 5 of course X."
            assert sources == ["Course Y - Lesson 1"]
            assert mock_client.chat.completions.create.call_count == 3
            assert mock_vector_store.search.call_count == 2
            mock_session_manager.add_exchange.assert_called_once()

    def test_api_query_wraps_backend_exceptions_as_generic_500(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        original_cwd = os.getcwd()

        try:
            os.chdir(backend_dir)

            import app as app_module

            mock_rag_system = MagicMock()
            mock_rag_system.session_manager.create_session.return_value = "session_1"
            mock_rag_system.query.side_effect = RuntimeError("backend exploded")
            app_module.rag_system = mock_rag_system

            client = TestClient(app_module.app)
            response = client.post("/api/query", json={"query": "test question"})

            assert response.status_code == 500
            assert response.json()["detail"] == "Query failed. Please try again."
        finally:
            os.chdir(original_cwd)
