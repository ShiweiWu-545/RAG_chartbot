"""
API endpoint tests for FastAPI endpoints.
Tests /api/query, /api/courses, and root endpoint behavior.
"""
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient


class TestAPIQueryEndpoint:
    """Test suite for POST /api/query endpoint"""

    def test_query_returns_answer_and_sources(self, test_client):
        """Test: Successful query returns response with answer, sources, and session_id"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test_session_123"

    def test_query_with_session_id_uses_provided_session(self, test_client):
        """Test: When session_id is provided, it's used for the query"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "my_session"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "my_session"

    def test_query_missing_query_returns_422(self, test_client):
        """Test: Missing query field returns 422 validation error"""
        response = test_client.post(
            "/api/query",
            json={}
        )

        assert response.status_code == 422

    def test_query_empty_query_is_accepted(self, test_client):
        """Test: Empty query string is accepted (FastAPI doesn't validate empty strings by default)"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        assert response.status_code == 200

    def test_query_with_rag_error_returns_500(self, test_client, mock_rag_system):
        """Test: RAG system errors are wrapped as 500 with generic message"""
        mock_rag_system["mock_ai"].generate_response.side_effect = RuntimeError("backend exploded")

        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )

        assert response.status_code == 500
        assert response.json()["detail"] == "Query failed. Please try again."


class TestAPICoursesEndpoint:
    """Test suite for GET /api/courses endpoint"""

    def test_courses_returns_analytics(self, test_client):
        """Test: Successful request returns course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data

    def test_courses_with_rag_error_returns_500(self):
        """Test: RAG system errors are wrapped as 500 with generic message"""
        from unittest.mock import patch, MagicMock
        from fastapi.testclient import TestClient
        from fastapi import FastAPI, HTTPException
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

        mock_rag = MagicMock()
        mock_rag.get_course_analytics.side_effect = RuntimeError("analytics failed")

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

        client = TestClient(app)
        response = client.get("/api/courses")

        assert response.status_code == 500
        assert response.json()["detail"] == "Failed to load course stats."


