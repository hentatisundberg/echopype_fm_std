from __future__ import annotations

from itertools import product
from typing import Iterable, Mapping

import numpy as np


def score_surface_prediction(
    reference: np.ndarray,
    prediction: np.ndarray,
    *,
    under_penalty: float = 10.0,
    over_penalty: float = 1.0,
    missing_penalty: float = 100.0,
) -> dict[str, float]:
    """Score a predicted range line, penalizing an unsafe shallow line more.

    Range increases downwards, so ``reference - prediction`` is positive when
    the detector stops above the annotated line and leaves unmasked pixels.
    """
    reference = np.asarray(reference, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction must have the same shape")
    if under_penalty < 0 or over_penalty < 0 or missing_penalty < 0:
        raise ValueError("penalties must be non-negative")

    labelled = np.isfinite(reference)
    if not labelled.any():
        raise ValueError("reference contains no finite annotations")
    errors = prediction[labelled] - reference[labelled]
    finite_prediction = np.isfinite(errors)
    under = np.maximum(-errors[finite_prediction], 0.0)
    over = np.maximum(errors[finite_prediction], 0.0)
    missing = float((~finite_prediction).sum())
    absolute = np.abs(errors[finite_prediction])
    total = errors.size
    weighted_loss = (
        float((under_penalty * under).sum())
        + float((over_penalty * over).sum())
        + missing_penalty * missing
    ) / total
    return {
        "score": weighted_loss,
        "labelled_pings": float(total),
        "detected_pings": float(finite_prediction.sum()),
        "missing_fraction": missing / total,
        "mae_m": float(absolute.mean()) if absolute.size else float("nan"),
        "rmse_m": float(np.sqrt(np.mean(np.square(errors[finite_prediction]))))
        if absolute.size
        else float("nan"),
        "mean_under_m": float(under.mean()) if under.size else 0.0,
        "mean_over_m": float(over.mean()) if over.size else 0.0,
        "max_under_m": float(under.max()) if under.size else 0.0,
        "under_fraction": float((under > 0).mean()) if finite_prediction.any() else 1.0,
        "over_fraction": float((over > 0).mean()) if finite_prediction.any() else 0.0,
    }


def parameter_combinations(
    base_params: Mapping[str, object],
    grid: Mapping[str, Iterable[object]],
) -> list[dict[str, object]]:
    """Return deterministic Cartesian products of parameter candidates."""
    names = list(grid)
    candidates = [list(grid[name]) for name in names]
    if any(not values for values in candidates):
        raise ValueError("each parameter grid must contain at least one value")
    return [
        {**base_params, **dict(zip(names, values))}
        for values in product(*candidates)
    ]
