from __future__ import annotations

import numpy as np

from echopype_fm_std.surface_optimization import (
    parameter_combinations,
    score_surface_prediction,
)


def test_score_penalizes_shallow_prediction_more_than_deep_prediction():
    reference = np.array([10.0, 10.0])
    shallow = score_surface_prediction(reference, [9.0, 10.0])
    deep = score_surface_prediction(reference, [11.0, 10.0])

    assert shallow["score"] == 5.0
    assert deep["score"] == 0.5
    assert shallow["under_fraction"] == 0.5


def test_score_penalizes_missing_predictions():
    metrics = score_surface_prediction([10.0, 10.0], [np.nan, 11.0])

    assert metrics["missing_fraction"] == 0.5
    assert metrics["detected_pings"] == 1
    assert metrics["score"] == 50.5


def test_parameter_combinations_override_base_values():
    combinations = parameter_combinations(
        {"threshold_db": -77.0, "fixed": 1},
        {"threshold_db": [-80.0, -75.0], "window": [3, 5]},
    )

    assert combinations == [
        {"threshold_db": -80.0, "fixed": 1, "window": 3},
        {"threshold_db": -80.0, "fixed": 1, "window": 5},
        {"threshold_db": -75.0, "fixed": 1, "window": 3},
        {"threshold_db": -75.0, "fixed": 1, "window": 5},
    ]
