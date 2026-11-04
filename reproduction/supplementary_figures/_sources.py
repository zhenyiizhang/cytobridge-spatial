"""Load the numerical result APIs and archived panel-drawing routines."""
from pathlib import Path
import importlib
import sys
import types
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
sys.path.insert(0,str(WORK))
sys.path.insert(0,str(ROOT/'code'))
ORIGINAL_SAVE=mpl.figure.Figure.savefig
for filename in ('Arial.ttf','Arial Bold.ttf','Arial Italic.ttf'):
    font=Path('/System/Library/Fonts/Supplemental')/filename
    if font.exists():mpl.font_manager.fontManager.addfont(font)
mpl.rcParams.update({'font.family':'Arial','font.size':9,'pdf.fonttype':42,
    'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'legend.frameon':False,'axes.grid':False,
    'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black'})

def module(name):
    return importlib.import_module('CytoBridge.results.'+name)

def plotter(name):
    path=ROOT/'code'/f'{name}.py'
    module_name='CytoBridge.results.publication_'+name
    result=types.ModuleType(module_name)
    result.__package__='CytoBridge.results'
    result.__file__=str(path)
    sys.modules[module_name]=result
    exec(compile(path.read_text(),str(path),'exec'),result.__dict__)
    return result
