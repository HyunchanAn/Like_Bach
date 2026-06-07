filepath = r'C:\Users\sg\.gemini\antigravity\brain\e8cdc7b6-66a4-4eac-aabf-9042302e4b20\task.md'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('[ ]', '[x]').replace('[/]', '[x]')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
