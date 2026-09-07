"""Recreate the September 6 supplementary figures from numerical results."""
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import fitz
import plot_panels as panels
import plot_summaries as summaries
import plot_domains as domains
from CytoBridge import zebrafish_attention_validation as attention

FIGURES=(2,3,4,5,6,7,8,25,34,36,39,40,41,42,43,44,45,46)
ROOT=Path(__file__).resolve().parent

def commot_permutations(output, results_dir=None, commot_results_dir=None):
    if commot_results_dir is not None:
        source=Path(commot_results_dir)
    elif results_dir is not None:
        source=Path(results_dir)/'commot_comparison'
    else:
        source=ROOT/'data/commot_comparison'
    left=pd.read_csv(source/'cytobridge_type_pair_summary.csv')
    right=pd.read_csv(source/'commot_type_pair_scores.csv.gz')
    score='abundance_controlled_distinct_cell_score'
    types=sorted(set(left.sender_type)|set(left.receiver_type)|set(right.sender_type)|set(right.receiver_type))
    external=attention.complete_directed_pair_table(right,score_column=score,cell_types=types)
    data=left.merge(external,on=list(attention.PAIR_KEYS),validate='one_to_one')
    design=attention._pair_covariate_matrix(data)
    x=data.G_AB_attention_mean_mean.to_numpy(float)
    y=attention._rank_residuals(data[score],design)
    observed=float(spearmanr(attention._rank_residuals(x,design),y).statistic)
    strata=attention.adaptive_pair_strata(data)
    groups=[np.flatnonzero(strata==g) for g in np.unique(strata)]
    rng=np.random.default_rng(20260816)
    null=np.empty(1000)
    for i in range(len(null)):
        shuffled=x.copy()
        for group in groups:
            if len(group)>1:shuffled[group]=x[rng.permutation(group)]
        null[i]=spearmanr(attention._rank_residuals(shuffled,design),y).statistic
    ref=panels.api('zebrafish_attention').load_zebrafish_attention_results(results_dir).panels.external_agreement
    ref=ref.set_index('external_method').loc['COMMOT']
    np.testing.assert_allclose(
        [observed,null.mean(),np.quantile(null,.025),np.quantile(null,.975)],
        [ref.adjusted_spearman_rho,ref.null_adjusted_spearman_mean,
         ref.null_adjusted_spearman_q025,ref.null_adjusted_spearman_q975],rtol=0,atol=1e-12)
    pd.DataFrame({'iteration':np.arange(1,1001),'adjusted_spearman_rho':null}).to_csv(
        output/'S39_within_group_permutations.csv',index=False)
    return null

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',default=[f'S{n}' for n in FIGURES])
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/supplementary_figures'))
    parser.add_argument('--results-dir',type=Path,help='Numerical input directory, instead of the included data')
    parser.add_argument('--commot-results-dir',type=Path,help='S39 CytoBridge and COMMOT pair-score directory')
    parser.add_argument('--calculated-inputs',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    requested=[int(n.upper().removeprefix('S')) for n in args.figures]
    calculated={}
    if args.calculated_inputs is not None:
        if not any(set(requested)<=group for group in ({2,3},{4,5},{25})):
            parser.error('Calculated objects require S2/S3, S4/S5 or S25')
        # Internal transport written by draw_supplementary, not a public data format.
        with args.calculated_inputs.open('rb') as handle:
            data,panel_values=pickle.load(handle)
        calculated={'data':data,'panel_values':panel_values}
    input_groups=({2,3},{4,5},{6},{25},{34,36},{39},{41},{42},{43},{44},{45},{46})
    if args.results_dir is not None and not any(set(requested)<=group for group in input_groups):
        parser.error('Requested figures must share a supported numerical input format')
    if args.commot_results_dir is not None and 39 not in requested:
        parser.error('--commot-results-dir is only used by S39')
    if any(n not in FIGURES for n in requested):parser.error('Choose figures from '+', '.join(f'S{n}' for n in FIGURES))
    output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    tables=output/'tables';tables.mkdir(exist_ok=True)
    for module in (panels,summaries,domains):
        module.OUT=output
        module.TABLES=tables
    domains.save=summaries.save
    calls={2:lambda:summaries.s2(args.results_dir,**calculated),3:lambda:panels.agist(args.results_dir,**calculated),
        4:lambda:panels.nonspatial(clone_values=True,figures=('s4',),results_dir=args.results_dir,**calculated),
        5:lambda:panels.nonspatial(figures=('s5',),results_dir=args.results_dir,**calculated),
        6:lambda:panels.classifier(args.results_dir),7:panels.heart,8:panels.heart,
        25:lambda:domains.s25(control_label='Randomly sampled cells',results_dir=args.results_dir,**calculated),
        34:lambda:panels.zebrafish(args.results_dir),36:lambda:panels.zebrafish(args.results_dir),
        39:lambda:panels.attention(results_dir=args.results_dir,
            permutation_values=commot_permutations(tables,args.results_dir,args.commot_results_dir)),
        40:summaries.s40,41:lambda:panels.lr(top_n=100,results_dir=args.results_dir),
        42:lambda:panels.ablation(results_dir=args.results_dir),
        43:lambda:panels.communication(args.results_dir),
        44:lambda:panels.wins(uniform_markers=True,results_dir=args.results_dir),
        45:lambda:panels.benchmark(args.results_dir),46:lambda:panels.training(args.results_dir)}
    done=set()
    for n in requested:
        if n in done:continue
        print(f'Drawing S{n}',flush=True)
        panels.defaults()
        # These three panels were approved together with this shared text scale.
        if n in (2,25,40):
            panels.mpl.rcParams.update({'axes.titlesize':10,'axes.labelsize':9,
                'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5})
        calls[n]()
        done.update((34,36) if n in (34,36) else ((7,8) if n in (7,8) else (n,)))
    combined=fitz.open()
    for n in sorted(set(requested)):
        with fitz.open(output/f'S{n}.pdf') as figure:
            combined.insert_pdf(figure)
    combined.set_toc([[1,f'S{n}',i+1] for i,n in enumerate(sorted(set(requested)))])
    combined.save(output/'supplementary_figures.pdf',garbage=4,deflate=True)
    combined.close()
    (output/'run.json').write_text(json.dumps({'figures':sorted(set(requested)),
        'source_directory':str(ROOT),'s2_summary':'median','s41_top_n':100,
        's39a':'within-group permutation histogram','s39_permutations':1000},indent=2))
    print(output/'supplementary_figures.pdf',flush=True)

if __name__=='__main__':main()
