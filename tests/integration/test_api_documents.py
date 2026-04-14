"""
Integration test: document upload and delete endpoints.
"""
import io
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
def test_upload_unsupported_extension(api_client, tmp_path):
    content = b"some content"
    resp = api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.xyz", io.BytesIO(content), "text/plain")},
        data={"user_id": "test"},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


@pytest.mark.integration
def test_upload_txt_file(api_client, sample_txt_file):
    mock_result = MagicMock()
    mock_result.document_id = "abc123"
    mock_result.source_path = str(sample_txt_file)
    mock_result.n_chunks = 3
    mock_result.collection_name = "aetheris"
    from datetime import datetime
    mock_result.ingested_at = datetime.utcnow()

    with patch("aetheris.api.routers.documents.ingest_file", return_value=mock_result):
        resp = api_client.post(
            "/api/v1/documents/upload",
            files={"file": ("sample.txt", sample_txt_file.read_bytes(), "text/plain")},
            data={"user_id": "test"},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert data["document_id"] == "abc123"
    assert data["n_chunks"] == 3


@pytest.mark.integration
def test_delete_nonexistent_document(api_client):
    with patch("aetheris.api.routers.documents.delete_document", return_value=0):
        resp = api_client.delete("/api/v1/documents/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.integration
def test_delete_existing_document(api_client):
    with patch("aetheris.api.routers.documents.delete_document", return_value=5):
        resp = api_client.delete("/api/v1/documents/existing-id")
    assert resp.status_code == 204
