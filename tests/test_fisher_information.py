from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sensing.fisher_information import (
    fim_metrics,
    fisher_information_matrix,
    measurement_jacobian,
    range_bearing_measurement,
    regularize_fim,
)


def test_range_bearing_measurement_and_jacobian_are_finite() -> None:
    target = np.array([220.0, 250.0, -120.0])
    sensor = np.array([200.0, 200.0, 0.0])

    measurement = range_bearing_measurement(target, sensor)
    jacobian = measurement_jacobian(target, sensor)

    assert measurement.shape == (2,)
    assert jacobian.shape == (2, 3)
    assert np.all(np.isfinite(measurement))
    assert np.all(np.isfinite(jacobian))


def test_fim_is_symmetric_and_regularized_psd() -> None:
    target = np.array([230.0, 220.0, -120.0])
    sensors = [
        np.array([200.0, 200.0, 100.0]),
        np.array([205.0, 190.0, 0.0]),
        np.array([215.0, 210.0, -120.0]),
    ]
    covariances = [
        np.diag([20.0**2, 0.15**2]),
        np.diag([10.0**2, 0.08**2]),
        np.diag([5.0**2, 0.04**2]),
    ]

    fim = fisher_information_matrix(target, sensors, covariances)
    regularized = regularize_fim(fim, regularization=1e-6)
    metrics = fim_metrics(fim, regularization=1e-6)

    assert fim.shape == (3, 3)
    assert np.allclose(fim, fim.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(regularized)) >= -1e-10
    assert np.isfinite(metrics["logdet"])
    assert np.isfinite(metrics["trace_inv"])
    assert metrics["trace_inv"] > 0.0
