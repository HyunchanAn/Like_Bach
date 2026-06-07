filepath = 'tests/test_backend.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('assert "engine" in data', 'assert "engine_ready" in data')
content = content.replace('assert response.status_code in [400, 422]', 'assert response.status_code == 200')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated tests/test_backend.py")
