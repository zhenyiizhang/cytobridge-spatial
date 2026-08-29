import json

notebook_path = '/lustre/home/2501111653/CytoBridge-ST-1104/evaluation/mosta_simulation_evaluation.ipynb'
output_path = 'extracted_evaluation.py'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                f.write(f"# Cell {i}\n")
                f.write(source)
                f.write("\n\n")
    print(f"Code extracted to {output_path}")
except Exception as e:
    print(f"Error extracting code: {e}")
