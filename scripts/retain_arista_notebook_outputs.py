"""Retain real execution outputs only when every code cell matches exactly."""
import argparse
from copy import deepcopy
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ['dataset_workflows/arista.ipynb', 'paper_figures/main_figure_5.ipynb',
         'paper_figures/arista_figures.ipynb', 'paper_figures/arista_local_domains.ipynb']


def retain(executed_dir):
    prepared = []
    for name in BOOKS:
        path = ROOT / 'docs/tutorials' / name
        current = nbformat.read(path, as_version=4)
        executed = nbformat.read(Path(executed_dir) / Path(name).name, as_version=4)
        old = [cell for cell in executed.cells if cell.cell_type == 'code']
        new = [cell for cell in current.cells if cell.cell_type == 'code']
        if [cell.source for cell in old] != [cell.source for cell in new]:
            raise ValueError(f'Code differs from real execution; execute again before retaining outputs: {name}')
        if any(cell.execution_count is None or any(o.output_type == 'error' for o in cell.outputs) for cell in old):
            raise ValueError(f'Execution is incomplete or contains errors: {name}')
        for actual, target in zip(old, new):
            target.outputs = deepcopy(actual.outputs)
            target.execution_count = actual.execution_count
            target.metadata = deepcopy(actual.metadata)
        prepared.append((path, current))
    for path, notebook in prepared:
        nbformat.write(notebook, path)
        print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executed-dir', type=Path, required=True)
    retain(parser.parse_args().executed_dir)
