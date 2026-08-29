"""Training entrypoints and orchestration."""

try:
    from .trainer import TrainingPipeline
except ModuleNotFoundError as exc:
    if exc.name in {"anndata"}:
        _training_pipeline_import_error = exc

        class TrainingPipeline:  # type: ignore[override]
            def __init__(self, *args, **kwargs):
                raise ModuleNotFoundError(
                    "CytoBridge.tl.train.TrainingPipeline requires optional dependency 'anndata'."
                ) from _training_pipeline_import_error
    else:
        raise

try:
    from .fit import fit, fit_spatial_csv, fit_spatial_h5ad
except ModuleNotFoundError as exc:
    if exc.name in {"scanpy", "anndata"}:
        _fit_import_error = exc

        def _missing_fit(*args, **kwargs):
            raise ModuleNotFoundError(
                "CytoBridge.tl.train.fit requires optional dependencies 'scanpy' and 'anndata'. "
                "Please install them to use training APIs."
            ) from _fit_import_error

        fit = _missing_fit
        fit_spatial_csv = _missing_fit
        fit_spatial_h5ad = _missing_fit
    else:
        raise

__all__ = [
    "fit",
    "fit_spatial_csv",
    "fit_spatial_h5ad",
    "TrainingPipeline",
]
