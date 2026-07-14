filepath = 'src/v5/neural_engine.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'from src.v4.models import BachTransformer, BachTokenizer\n',
    'from src.v4.models import BachTransformer, BachTokenizer\nimport src.v4.models as models\nimport sys\nsys.modules["models"] = models\nsys.modules["src.models"] = models\n'
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated src/v5/neural_engine.py with sys.modules mock for pickle")
