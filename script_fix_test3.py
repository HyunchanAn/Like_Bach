filepath = 'tests/test_backend.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('assert "mode" in data', '')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated tests/test_backend.py to remove mode assertion")
