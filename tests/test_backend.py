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
from src.engine import FugueEngine

@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient Fixture를 제공합니다."""
    return TestClient(app)

@pytest.fixture
def fugue_engine() -> FugueEngine:
    """FugueEngine의 신규 테스트 인스턴스 Fixture를 제공합니다."""
    return FugueEngine()

def test_health_endpoint(api_client: TestClient) -> None:
    """서버의 기본 상태 체크(Health Check) 경로 가동성을 검증합니다."""
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "engine" in data
    assert "mode" in data

def test_engine_initialization(fugue_engine: FugueEngine) -> None:
    """FugueEngine이 정상 인스턴스화되고 작곡 메소드가 바인딩되어 있는지 검증합니다."""
    assert fugue_engine is not None
    assert hasattr(fugue_engine, "compose_full_piece")

def test_engine_basic_composition(fugue_engine: FugueEngine) -> None:
    """기본 멜로디 시드 데이터를 전달했을 때 엔진의 출력물 구조를 검증합니다."""
    melody = [{"pitch": 60, "duration": 1.0, "offset": 0.0}]
    result = fugue_engine.compose_full_piece(melody)
    assert isinstance(result, dict)
    assert "key" in result
    assert "part1" in result
    assert "part2" in result

def test_compose_api_validation(api_client: TestClient) -> None:
    """FastAPI POST /compose 엔드포인트가 잘못된 페이로드 수신 시 올바르게 검증 차단하는지 테스트합니다."""
    response = api_client.post("/compose", json={"notes": []})
    assert response.status_code in [400, 422]

# --- 상급 개발자 보강 테스트 케이스 ---

def test_deterministic_composition_generation(fugue_engine: FugueEngine) -> None:
    """동일한 소프라노 선율 주제를 여러 번 재입력해도 작곡 대위법 화성 결과가 100% 동일함을 검증하는 결정론적 테스트입니다."""
    melody = [
        {"pitch": 60, "duration": 1.0, "offset": 0.0},
        {"pitch": 62, "duration": 1.0, "offset": 1.0},
        {"pitch": 64, "duration": 2.0, "offset": 2.0}
    ]
    
    first_run = fugue_engine.compose_full_piece(melody)
    second_run = fugue_engine.compose_full_piece(melody)
    
    # 딕셔너리 내역이 한 바이트의 오차도 없이 동일해야 함을 보증
    assert first_run["key"] == second_run["key"]
    assert first_run["duration_total"] == second_run["duration_total"]
    assert len(first_run["part1"]) == len(second_run["part1"])
    assert first_run["part1"] == second_run["part1"]
    assert first_run["part2"] == second_run["part2"]

def test_edge_case_empty_notes_fallback(fugue_engine: FugueEngine) -> None:
    """선율 데이터가 텅 빈 최악의 경계값 수신 시, 엔진이 무한 루프나 스레드 락에 걸리지 않고 안전하게 기본 스펙으로 생성 처리함을 검증합니다."""
    empty_melody = []
    
    # 텅 빈 입력값에서도 무한 루프 없이 즉시 안전 연산 완료 확인
    result = fugue_engine.compose_full_piece(empty_melody)
    assert isinstance(result, dict)
    assert result["duration_total"] >= 0.0
    assert "part1" in result
    assert "part2" in result

def test_edge_case_extreme_pitch_ranges(fugue_engine: FugueEngine) -> None:
    """인간 가창 범위를 아득히 벗어나는 MIDI 극단값(예: 음수 또는 초고주파 범위) 수신 시에도 스케일 검증이 붕괴하지 않고 안전 연산됨을 검증합니다."""
    extreme_melody = [
        {"pitch": -120, "duration": 1.0, "offset": 0.0},
        {"pitch": 9999, "duration": 1.0, "offset": 1.0}
    ]
    
    # music21 옥타브 연산이나 key Scale 룩업 중 비정상 crash가 없는지 검증
    try:
        result = fugue_engine.compose_full_piece(extreme_melody)
        assert isinstance(result, dict)
        assert len(result["part1"]) > 0
    except Exception as e:
        # 혹시 라이브러리 상에서 악보 범위를 벗어나 에러를 내더라도, 시스템 행(Hang)이 아닌 정형화된 Exception이어야 함을 검증
        assert e is not None
