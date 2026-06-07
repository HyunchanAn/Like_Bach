filepath = 'src/v5/neural_engine.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('except:', 'except Exception:')
content = content.replace('print(f"Dual Neural Engine (Chorale & Fugue) loaded.")', 'print("Dual Neural Engine (Chorale & Fugue) loaded.")')
content = content.replace('add_debug(curr_measure, f"-> [V1] 대주제(CS1) AI 생성 호출")', 'add_debug(curr_measure, "-> [V1] 대주제(CS1) AI 생성 호출")')
content = content.replace('add_debug(curr_measure, f"-> [V1, V2] 대주제(CS2, CS1) AI 생성 호출")', 'add_debug(curr_measure, "-> [V1, V2] 대주제(CS2, CS1) AI 생성 호출")')
content = content.replace('add_debug(curr_measure, f"-> [V1, V2, V3] 대주제(CS3, CS2, CS1) AI 생성 호출")', 'add_debug(curr_measure, "-> [V1, V2, V3] 대주제(CS3, CS2, CS1) AI 생성 호출")')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated src/v5/neural_engine.py")
