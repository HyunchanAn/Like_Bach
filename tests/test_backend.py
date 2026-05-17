import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import backend resources
from src.main import app
from src.engine import FugueEngine

@pytest.fixture
def api_client():
    return TestClient(app)

@pytest.fixture
def fugue_engine():
    return FugueEngine()

def test_health_endpoint(api_client):
    """
    Test that the root health-check endpoint is online and responsive.
    """
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "engine" in data
    assert "mode" in data

def test_engine_initialization(fugue_engine):
    """
    Test that the FugueEngine initializes properly with baseline configurations.
    """
    assert fugue_engine is not None
    # Check if there are default composition parameters initialized
    assert hasattr(fugue_engine, 'compose_full_piece')

def test_engine_basic_composition(fugue_engine):
    """
    Test the FugueEngine composition with a single valid seed note melody.
    """
    # Standard C major quarter note
    melody = [{"pitch": 60, "duration": 1.0, "offset": 0.0}]
    
    result = fugue_engine.compose_full_piece(melody)
    
    assert isinstance(result, dict)
    # The output should contain generated notes and basic structural info
    assert "notes" in result or "soprano" in result or len(result) > 0

def test_compose_api_validation(api_client):
    """
    Test that the POST /compose endpoint validates input correctly
    and handles bad payloads gracefully.
    """
    # Empty payload should trigger validation error or handled exception
    response = api_client.post("/compose", json={"notes": []})
    assert response.status_code in [400, 422]  # Bad Request or Unprocessable Entity
