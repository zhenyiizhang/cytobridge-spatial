"""Draw supplementary figures using the figure numbers in the current paper."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
NUMBERING=json.loads((ROOT/'reproduction/figure_numbering.json').read_text())

def draw(figures, output_dir, **kwargs):
    from reproduction.paper_figures import draw_supplementary
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    numbers=[int(n) for n in figures]
    mapping=NUMBERING['current_to_archive']; result={}
    if (kwargs.get("data") is None) != (kwargs.get("panels") is None):
        raise ValueError("Provide both data and panels together")
    source = kwargs.get("results_dir")
    if source is None and kwargs.get("data") is not None:
        source = kwargs["data"].source_dir
    if source is not None:
        source = Path(source).expanduser().resolve()
        if output == source or source in output.parents:
            raise ValueError("Choose an output directory outside the input directory")
    if 2 in numbers:
        raise ValueError("Recalculate S2 with tutorials/paper_figures/main_figure_2.ipynb")
    legacy=[mapping[str(n)] for n in numbers if n!=1]
    if legacy:
        with tempfile.TemporaryDirectory(prefix='cytobridge-draw-') as tmp:
            try:
                paths=draw_supplementary(legacy,tmp,**kwargs)
            except ValueError:
                raise
            except Exception:
                log=Path(tmp)/'plotting.log'
                if log.exists():shutil.copy2(log,output/'plotting.log')
                raise RuntimeError(f'Figure drawing failed. See {output / "plotting.log"}') from None
            for n in numbers:
                if n==1:continue
                pdf,png=paths[f's{mapping[str(n)]}']
                for source in (pdf,png):shutil.copy2(source,output/f'S{n}{source.suffix}')
            shutil.copytree(Path(tmp)/'tables',output/'tables',dirs_exist_ok=True)
    if 1 in numbers:
        for suffix in ['pdf','png']:shutil.copy2(ROOT/f'reproduction/workflow/S1_workflow.{suffix}',output/f'S1.{suffix}')
    for n in numbers:result[f's{n}']=(output/f'S{n}.pdf',output/f'S{n}.png')
    return result

draw_supplementary = draw

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--figures',type=int,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--results-dir',type=Path)
    a=p.parse_args()
    for n,paths in draw(a.figures,a.output_dir,results_dir=a.results_dir).items():print(n.upper(),*paths)
