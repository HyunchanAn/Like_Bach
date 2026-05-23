import os
import sys
import pytest
import music21
from fastapi.testclient import TestClient

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import backend resources
from src.main import app

@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient Fixture를 제공합니다."""
    return TestClient(app)

def test_health_endpoint(api_client: TestClient) -> None:
    """서버의 기본 상태 체크(Health Check) 경로 가동성을 검증합니다."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "engine" in data
    assert "mode" in data

def test_compose_api_validation(api_client: TestClient) -> None:
    """FastAPI POST /compose 엔드포인트가 잘못된 페이로드 수신 시 올바르게 검증 차단하는지 테스트합니다."""
    response = api_client.post("/compose", json={"notes": []})
    assert response.status_code in [400, 422]
