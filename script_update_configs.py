def replace_in_file(filepath, old_str, new_str):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {filepath}")

replace_in_file('ui/v4-app/vite.config.ts', 'src/v4/api.py', 'src/v5/api.py')
replace_in_file('README.md', 'src/v4/api.py', 'src/v5/api.py')
replace_in_file('README.md', 'src.v4.api:app', 'src.v5.api:app')
replace_in_file('initial.bat', 'src/v4/api.py', 'src/v5/api.py')
replace_in_file('tests/test_backend.py', 'from src.main import app', 'from src.v5.api import app')
replace_in_file('tests/test_backend.py', 'from src.v4.api import app', 'from src.v5.api import app')
