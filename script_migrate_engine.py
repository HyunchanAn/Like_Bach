import os

filepath_v4 = 'src/v4/neural_engine.py'
filepath_v5 = 'src/v5/neural_engine.py'

with open(filepath_v4, 'r', encoding='utf-8') as f:
    lines_v4 = f.readlines()
    
# Extract NeuralBachEngine class
class_start = -1
for i, line in enumerate(lines_v4):
    if line.startswith('class NeuralBachEngine:'):
        class_start = i
        break

if class_start != -1:
    class_code = "".join(lines_v4[class_start:])
    
    with open(filepath_v5, 'r', encoding='utf-8') as f:
        content_v5 = f.read()
        
    # Append class to v5
    if 'class NeuralBachEngine:' not in content_v5:
        # Add import for BachTransformer and BachTokenizer at the top
        import_stmt = "\nfrom src.v4.models import BachTransformer, BachTokenizer\n"
        
        # Find where to insert import
        lines_v5 = content_v5.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines_v5):
            if line.startswith('from src.v5.models'):
                insert_idx = i + 1
                break
                
        lines_v5.insert(insert_idx, import_stmt)
        
        content_v5 = "\n".join(lines_v5) + "\n\n" + class_code
        
        with open(filepath_v5, 'w', encoding='utf-8') as f:
            f.write(content_v5)
        print("NeuralBachEngine successfully migrated to v5/neural_engine.py")
    else:
        print("NeuralBachEngine already exists in v5/neural_engine.py")
else:
    print("Could not find NeuralBachEngine in v4/neural_engine.py")
