filepath = 'tests/test_backend.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('data["status"] == "online"', 'data["status"] == "ok"')

old_fixture = '''@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient Fixture를 제공합니다."""
    return TestClient(app)'''

new_fixture = '''@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient Fixture를 제공합니다."""
    with TestClient(app) as client:
        yield client'''

content = content.replace(old_fixture, new_fixture)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated tests/test_backend.py to use context manager and 'ok' status")
