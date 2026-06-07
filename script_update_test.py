filepath = 'tests/test_backend.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('response = api_client.get("/")', 'response = api_client.get("/api/health")')
content = content.replace('response = api_client.post("/compose", json={"notes": []})', 'response = api_client.post("/api/generate", json={"subject_notes": []})')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated tests/test_backend.py")
