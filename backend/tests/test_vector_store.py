"""
Tests for VectorStore embedding initialization and offline fallback behavior.
"""
import sys
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import LocalHashEmbeddingFunction, VectorStore


class TestLocalHashEmbeddingFunction:
    def test_returns_fixed_size_embeddings(self):
        embedder = LocalHashEmbeddingFunction(dimension=16)

        embeddings = embedder(["Python basics", "Machine learning basics"])

        assert len(embeddings) == 2
        assert embeddings[0].shape == (16,)
        assert embeddings[1].shape == (16,)

    def test_is_deterministic_for_same_input(self):
        embedder = LocalHashEmbeddingFunction(dimension=16)

        first = embedder(["Lesson summary"])[0]
        second = embedder(["Lesson summary"])[0]

        assert np.allclose(first, second)


class TestVectorStoreEmbeddingInitialization:
    def test_falls_back_to_local_embeddings_when_transformer_load_fails(self):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        with patch("vector_store.chromadb.PersistentClient", return_value=mock_client), \
             patch(
                 "vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
                 side_effect=RuntimeError("offline")
             ):
            store = VectorStore("./test_chroma_db", "all-MiniLM-L6-v2")

        assert isinstance(store.embedding_function, LocalHashEmbeddingFunction)
        assert mock_client.get_or_create_collection.call_count == 2
