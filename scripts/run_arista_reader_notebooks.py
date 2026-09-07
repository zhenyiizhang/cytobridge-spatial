"""Execute the four connected ARISTA notebooks in separate fresh kernels."""
import argparse
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ['dataset_workflows/arista.ipynb', 'paper_figures/main_figure_5.ipynb',
         'paper_figures/arista_figures.ipynb', 'paper_figures/arista_local_domains.ipynb']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, required=True)
    parser.add_argument('--analysis-dir', type=Path, required=True)
    parser.add_argument('--execution-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--notebooks', nargs='+', choices=BOOKS, default=BOOKS)
    args = parser.parse_args()
    os.environ.update(CYTOBRIDGE_PROJECT_DIR=str(args.project_dir.resolve()),
                      CYTOBRIDGE_ARISTA_OUTPUT_DIR=str(args.analysis_dir.resolve()),
                      CYTOBRIDGE_DEVICE=args.device, MPLBACKEND='Agg')
    args.execution_dir.mkdir(parents=True, exist_ok=True)
    for name in args.notebooks:
        print('EXECUTING', name, flush=True)
        notebook = nbformat.read(ROOT / 'docs/tutorials' / name, as_version=4)
        try:
            NotebookClient(notebook, timeout=7200, kernel_name='python3',
                           resources={'metadata': {'path': str(ROOT)}}).execute()
        finally:
            nbformat.write(notebook, args.execution_dir / Path(name).name)
        print('PASS', name, flush=True)


if __name__ == '__main__':
    main()
