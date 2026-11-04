"""Train an S4/S5 model with its distributed training implementation."""
from pathlib import Path
import argparse
import json
import math
import sys

import yaml

MODELS = {
    "weinreb_radius": "weinreb_radius",
    "weinreb_lr": "weinreb_lr",
    "scnt_lr": "scnt_lr",
    "weinreb_no_interaction": "no_interaction",
    "scnt_no_interaction": "no_interaction",
}


def configure_interaction(config, prepared_h5ad, edge_prior):
    """Use the cutoff and LR threshold fitted during input preparation."""
    import anndata as ad
    interaction = config["model"].get("interaction_net", {})
    if not interaction:
        return
    data = ad.read_h5ad(prepared_h5ad, backed="r")
    try:
        cutoff = float(data.uns.get("fit_params", {}).get("interaction_cutoff", float("nan")))
    finally:
        data.file.close()
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Prepared H5AD needs a positive fit_params.interaction_cutoff")
    interaction["cutoff"] = cutoff
    if interaction.get("edge_mode") == "predictor":
        prior = Path(edge_prior).resolve()
        manifest = json.loads((prior.parent / "manifest.json").read_text())
        threshold = float(manifest["predictor"]["recommended_edge_predictor_threshold"])
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("The fitted LR threshold must lie between zero and one")
        interaction["edge_predictor_path"] = str(prior)
        interaction["edge_predictor_thre"] = threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-prior", type=Path)
    parser.add_argument("--training-code", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.training_code
    if root is None:
        root = Path(__file__).resolve().parent
        if not (root / "runtimes").is_dir():
            root = Path("data/nonspatial/training_code")
    root = root.resolve()
    runtime = root / "runtimes" / MODELS[args.model]
    if not (runtime / "CytoBridge").is_dir():
        raise FileNotFoundError(
            "Download nonspatial_training_code.zip, then supply its extracted "
            "folder with --training-code."
        )
    if not args.input_h5ad.is_file():
        raise FileNotFoundError(args.input_h5ad)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new model directory: {output}")
    config_path = args.config or root / "configs" / f"{args.model}.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["ckpt_dir"] = str(output)
    if args.seed is not None:
        config["seed"] = args.seed
    interaction = config["model"].get("interaction_net", {})
    if interaction.get("edge_mode") == "predictor":
        if args.edge_prior is None or not args.edge_prior.is_file():
            parser.error("LR models require --edge-prior pointing to link_predictor.pt")
        interaction["edge_predictor_path"] = str(args.edge_prior.resolve())
    elif args.edge_prior is not None:
        parser.error("This model does not use an LR edge predictor")
    configure_interaction(config, args.input_h5ad, args.edge_prior)
    # Import the selected implementation before importing the installed package.
    sys.path.insert(0, str(runtime))
    from CytoBridge.tl.train import fit
    import CytoBridge
    if not Path(CytoBridge.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError("Run this training command in a new Python process")
    fit(str(args.input_h5ad.resolve()), config=config, device=args.device,
        time_key="time_point_processed", obsm_key="X_latent", is_spatial=False,
        interaction_cutoff=interaction.get("cutoff"), ckpt_dir=output,
        sigma=float(config["training"]["defaults"].get("sigma", .1)),
        evaluate_after_training=False)


if __name__ == "__main__":
    main()
