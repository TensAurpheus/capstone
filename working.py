import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if __package__:
    from .model_training import prepare_data, train_and_predict
else:  # Allow running as ``python src/apply_models.py``
    sys.path.append(str(Path(__file__).resolve().parent))
    from model_training import prepare_data, train_and_predict  # type: ignore

DEFAULT_MODELS: List[str] = ["catboost",
                             "lightgbm", "xgboost", "random_forest"]


def _normalise_models(models: Iterable[str]) -> List[str]:
    """Return lowercase model names without duplicates while preserving order."""

    seen = set()
    normalised: List[str] = []
    for name in models:
        key = name.lower()
        if key not in seen:
            normalised.append(key)
            seen.add(key)
    return normalised


def apply_models(
    parquet_path: str | Path,
    models: Optional[Iterable[str]] = None,
    *,
    progress: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Train each requested model and print validation/test metrics."""

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    model_names = _normalise_models(models or DEFAULT_MODELS)
    if not model_names:
        raise ValueError("At least one model name must be provided")

    print(f"Loading data from {path}...")
    data = prepare_data(str(path))
    print(
        "Loaded data with",
        len(data["train"]["X"]),
        "training rows and",
        len(data["feature_cols"]),
        "features.",
    )

    results: Dict[str, Dict[str, object]] = {}
    for model_name in model_names:
        print(f"\n=== Training {model_name} ===")
        result = train_and_predict(
            model_name,
            data,
            predict=True,
            progress=progress,
        )
        results[model_name] = result

        metrics = result.get("metrics")
        if metrics:
            for split_name, split_metrics in metrics.items():
                formatted = ", ".join(
                    f"{metric}={value:.4f}" for metric, value in split_metrics.items()
                )
                print(f"{model_name} {split_name} metrics: {formatted}")

        feature_importances = result.get("feature_importances")
        if feature_importances:
            print(
                f"Captured feature importances for {model_name} with"
                f" {len(feature_importances)} features."
            )

    return results


if __name__ == "__main__":
    # Update the path below to point to the desired preprocessed parquet file.
    default_parquet = Path("data/BTC_merged_15m_8h.parquet")
    apply_models(default_parquet)
