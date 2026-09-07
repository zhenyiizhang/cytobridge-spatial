"""Draw the current supplementary figures from the paper's numerical results."""

from pathlib import Path
import pickle
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = (2, 3, 4, 5, 6, 7, 8, 25, 34, 36, 39, 40, 41, 42, 43, 44, 45, 46)
INPUT_GROUPS = ({2, 3}, {4, 5}, {6}, {25}, {34, 36}, {39}, {41}, {42}, {43}, {44}, {45}, {46})


def draw_supplementary(figures, output_dir, *, results_dir=None, commot_results_dir=None,
                       data=None, panels=None):
    """Calculate summaries and draw the accepted SI panels from source tables.

    Parameters
    ----------
    figures : sequence of int
        Supplementary figure numbers, for example ``[4, 5]``.
    output_dir : path
        Directory for newly drawn PDF/PNG figures and their ``tables/`` folder.
    results_dir : path, optional
        Read this directory instead of the included inputs.
        It uses the corresponding ``CytoBridge.results`` loader format.
        S43 reads its four communication panel tables; S44 reads
        ``loto_target_stage_means_with_spatrack.csv`` from this directory.
    commot_results_dir : path, optional
        S39 pair-score directory containing ``cytobridge_type_pair_summary.csv``
        and ``commot_type_pair_scores.csv.gz``. When selecting new S39 results,
        provide this directory or include it as ``results_dir/commot_comparison``.
    data, panels : result objects, optional
        For S2/S3, S4/S5 or S25, pass both the loaded data and the panel tables
        calculated from it. The plotter consumes these exact values without
        loading or calculating them again. ``results_dir`` remains the input
        location for overwrite protection.

    Returns
    -------
    dict
        Figure IDs mapped to ``(pdf_path, png_path)``. The code runs the same
        numerical plotting programs used for the current SI. It neither reads
        completed figures nor trains a model. The input files are listed in
        ``reproduction/supplementary_figures/README.md``.
    """
    numbers = tuple(int(number) for number in figures)
    if not numbers or len(numbers) != len(set(numbers)) or set(numbers) - set(SUPPORTED):
        raise ValueError(f"Choose distinct figure numbers from {SUPPORTED}.")
    if (data is None) != (panels is None):
        raise ValueError("Pass both data and panels when plotting calculated values.")
    if data is not None:
        if not any(set(numbers) <= group for group in ({2, 3}, {4, 5}, {25})):
            raise ValueError("Calculated objects are supported for S2/S3, S4/S5 or S25.")
        source_dir = Path(data.source_dir).expanduser().resolve()
        if results_dir is not None and Path(results_dir).expanduser().resolve() != source_dir:
            raise ValueError("results_dir must match the supplied data.source_dir.")
        results_dir = source_dir
    output = Path(output_dir).expanduser().resolve()
    if results_dir is not None:
        if not any(set(numbers) <= group for group in INPUT_GROUPS):
            raise ValueError("Use figures sharing one input format: S2/S3, S4/S5, S6, S25, S34/S36, S39, S41, S42, S43, S44, S45 or S46.")
        results_dir = Path(results_dir).expanduser().resolve()
        if output == results_dir or results_dir in output.parents:
            raise ValueError("Save the figure outside the input directory.")
    if commot_results_dir is not None:
        if 39 not in numbers:
            raise ValueError("commot_results_dir is only used by S39.")
        commot_results_dir = Path(commot_results_dir).expanduser().resolve()
        if output == commot_results_dir or commot_results_dir in output.parents:
            raise ValueError("Save the figure outside the COMMOT input directory.")
    protected = [ROOT / p for p in ("CytoBridge", "reproduction", "release_artifacts")]
    if output == ROOT or any(output == p or p in output.parents for p in protected):
        raise ValueError("Write figures outside the code and input directories, for example outputs/figures.")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "reproduction/supplementary_figures/plot_figures.py"),
               "--figures", *(f"S{n}" for n in numbers), "--output-dir", str(output)]
    if results_dir is not None:
        command.extend(["--results-dir", str(results_dir)])
    if commot_results_dir is not None:
        command.extend(["--commot-results-dir", str(commot_results_dir)])
    # The accepted plotters use different Matplotlib settings. A separate
    # process leaves the reader's notebook settings and figures unchanged.
    with tempfile.TemporaryDirectory(prefix="cytobridge-calculated-panels-") as temporary, \
            (output / "plotting.log").open("w") as log:
        if data is not None:
            # This private file transports this process's trusted objects to
            # its own plotting subprocess; it is removed when drawing ends.
            calculated = Path(temporary) / "calculated_inputs.pkl"
            with calculated.open("wb") as handle:
                pickle.dump((data, panels), handle, protocol=pickle.HIGHEST_PROTOCOL)
            command.extend(["--calculated-inputs", str(calculated)])
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Figure calculation failed. See {output / 'plotting.log'}")
    result = {f"s{n}": (output / f"S{n}.pdf", output / f"S{n}.png") for n in numbers}
    missing = [str(p) for paths in result.values() for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"The plotting program did not produce: {missing}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=int, nargs="+", default=SUPPORTED)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/supplementary_figures"))
    parser.add_argument("--results-dir", type=Path, help="Numerical input directory for one supported input format")
    parser.add_argument("--commot-results-dir", type=Path, help="S39 CytoBridge and COMMOT pair-score directory")
    args = parser.parse_args()
    for name, paths in draw_supplementary(args.figures, args.output_dir, results_dir=args.results_dir,
                                        commot_results_dir=args.commot_results_dir).items():
        print(name.upper(), *(str(path) for path in paths))
