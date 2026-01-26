"""
Integration tests for RAG API.
Requires running server: docker compose up -d
"""

import requests
import pytest
import time

BASE_URL = "http://localhost:8000"


def wait_for_server(timeout: int = 30) -> bool:
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


class TestHealthEndpoints:
    """Test health and status endpoints."""

    def test_health_check(self):
        """Test /health endpoint."""
        r = requests.get(f"{BASE_URL}/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["components"]["rag_engine"] is True
        assert data["components"]["openai_configured"] is True

    def test_stats(self):
        """Test /stats endpoint."""
        r = requests.get(f"{BASE_URL}/stats")
        assert r.status_code == 200
        data = r.json()
        assert "indexed_documents" in data
        assert "model_name" in data
        assert "embedding_model" in data

    def test_root(self):
        """Test root endpoint."""
        r = requests.get(f"{BASE_URL}/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "LlamaHub + LightRAG API"
        assert "endpoints" in data


class TestSourcesEndpoints:
    """Test sources management endpoints."""

    def test_list_sources(self):
        """Test /sources endpoint."""
        r = requests.get(f"{BASE_URL}/sources")
        assert r.status_code == 200
        data = r.json()
        assert "sources" in data
        assert data["total"] >= 1

        # Check local_files source exists
        source_types = [s["source_type"] for s in data["sources"]]
        assert "local_files" in source_types

    def test_list_local_files(self):
        """Test /sources/local/files endpoint."""
        r = requests.get(f"{BASE_URL}/sources/local/files")
        assert r.status_code == 200
        data = r.json()
        assert "files" in data
        assert "total" in data
        assert "base_path" in data


class TestQueryModes:
    """Test query mode listing."""

    def test_list_query_modes(self):
        """Test /query/modes endpoint."""
        r = requests.get(f"{BASE_URL}/query/modes")
        assert r.status_code == 200
        data = r.json()
        assert "modes" in data

        mode_names = [m["name"] for m in data["modes"]]
        assert "naive" in mode_names
        assert "local" in mode_names
        assert "global" in mode_names
        assert "hybrid" in mode_names


class TestIndexing:
    """Test indexing endpoints."""

    def test_index_local_files(self):
        """Test /index/local endpoint with test_pdfs folder."""
        r = requests.post(
            f"{BASE_URL}/index/local",
            json={"path": "/app/test_pdfs", "recursive": True},
            timeout=300
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True
        assert data["source_type"] == "local_files"
        assert data["documents_indexed"] >= 0


class TestRAGQuery:
    """Test RAG query functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Ensure documents are indexed before queries."""
        # Index test documents
        requests.post(
            f"{BASE_URL}/index/local",
            json={"path": "/app/test_pdfs", "recursive": True},
            timeout=300
        )
        time.sleep(2)  # Wait for indexing

    def test_query_hybrid_mode(self):
        """Test query with hybrid mode (default)."""
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": "LightRAG란 무엇인가요?",
                "mode": "hybrid"
            },
            timeout=120
        )
        assert r.status_code == 200
        data = r.json()
        assert "answer" in data
        assert data["mode"] == "hybrid"
        assert len(data["answer"]) > 0
        assert "processing_time_ms" in data

    def test_query_local_mode(self):
        """Test query with local mode."""
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": "What is dual-level retrieval?",
                "mode": "local"
            },
            timeout=120
        )
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "local"
        assert len(data["answer"]) > 0

    def test_query_global_mode(self):
        """Test query with global mode."""
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": "What is incremental update algorithm?",
                "mode": "global"
            },
            timeout=120
        )
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "global"
        assert len(data["answer"]) > 0

    def test_query_naive_mode(self):
        """Test query with naive mode."""
        r = requests.post(
            f"{BASE_URL}/query",
            json={
                "question": "LightRAG features",
                "mode": "naive"
            },
            timeout=120
        )
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "naive"


if __name__ == "__main__":
    print("Waiting for server...")
    if not wait_for_server():
        print("Server not available. Start with: docker compose up -d")
        exit(1)

    print("Running tests...")
    pytest.main([__file__, "-v"])
