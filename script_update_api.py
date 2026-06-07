filepath = 'src/v5/api.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'from src.v4.neural_engine import NeuralBachEngine\nfrom src.v5.neural_engine import HybridFugueEngine',
    'from src.v5.neural_engine import NeuralBachEngine, HybridFugueEngine'
)
content = content.replace(
    'FastAPI(title="Like Bach v4.5 API Engine")',
    'FastAPI(title="Like Bach v5 API Engine")'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated src/v5/api.py")
