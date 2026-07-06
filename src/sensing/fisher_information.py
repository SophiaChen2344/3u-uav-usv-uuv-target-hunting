"""Fisher Information Matrix utilities for noisy 3U target sensing.

The paper-level system model assumes cooperative UAV-USV-UUV sensing, but the
clean simulator keeps the sensing model intentionally compact. Each available
platform contributes a range-bearing measurement of the target position:

``range = ||target - sensor||``
``bearing = atan2(y_target - y_sensor, x_target - x_sensor)``

The target state in the larger project may include speed and heading, but this
module evaluates information for the 3D target position only. The resulting FIM
therefore has shape ``(3, 3)``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12


def range_bearing_measurement(target_pos: Sequence[float], sensor_pos: Sequence[float]) -> np.ndarray:
    """Return ``[range, bearing]`` from ``sensor_pos`` to ``target_pos``.

    Range is computed in 3D, while bearing is the horizontal azimuth angle in
    radians. This simple model is shared by UAV, USV, and UUV-center sensors.
    """

    target = _as_position(target_pos)
    sensor = _as_position(sensor_pos)
    delta = target - sensor
    range_value = float(np.linalg.norm(delta))
    bearing = float(np.arctan2(delta[1], delta[0]))
    return np.array([range_value, bearing], dtype=float)


def measurement_jacobian(target_pos: Sequence[float], sensor_pos: Sequence[float]) -> np.ndarray:
    """Return the Jacobian of the range-bearing measurement w.r.t. target XYZ.

    The Jacobian has shape ``(2, 3)``. Near zero distance the derivatives are
    safely damped to avoid numerical divisions by zero.
    """

    target = _as_position(target_pos)
    sensor = _as_position(sensor_pos)
    dx, dy, dz = target - sensor
    range_norm = max(float(np.sqrt(dx * dx + dy * dy + dz * dz)), EPS)
    horizontal_sq = max(float(dx * dx + dy * dy), EPS)

    jacobian = np.zeros((2, 3), dtype=float)
    jacobian[0, :] = np.array([dx, dy, dz], dtype=float) / range_norm
    jacobian[1, 0] = -dy / horizontal_sq
    jacobian[1, 1] = dx / horizontal_sq
    return jacobian


def fisher_information_matrix(
    target_pos: Sequence[float],
    sensor_positions: Iterable[Sequence[float]],
    noise_covariances: Iterable[Sequence[Sequence[float]] | Sequence[float] | float],
) -> np.ndarray:
    """Assemble the 3D target-position FIM from independent sensors.

    ``noise_covariances`` contains one 2x2 covariance matrix per sensor for the
    range-bearing measurement. Scalars are interpreted as a shared variance for
    both measurement channels; a length-2 vector is interpreted as diagonal
    variances.
    """

    target = _as_position(target_pos)
    fim = np.zeros((3, 3), dtype=float)

    for sensor_pos, covariance in zip(sensor_positions, noise_covariances):
        sensor = _as_position(sensor_pos)
        jacobian = measurement_jacobian(target, sensor)
        covariance_matrix = _as_covariance(covariance)
        inv_covariance = np.linalg.pinv(covariance_matrix)
        fim += jacobian.T @ inv_covariance @ jacobian

    fim = 0.5 * (fim + fim.T)
    return np.nan_to_num(fim, nan=0.0, posinf=0.0, neginf=0.0)


def fim_metrics(FIM: Sequence[Sequence[float]], regularization: float = 1e-6) -> dict[str, float]:
    """Return scalar information metrics for a regularized FIM.

    The regularization term represents a weak prior and makes singular or
    poorly conditioned geometries numerically evaluable.
    """

    matrix = regularize_fim(FIM, regularization=regularization)
    eigenvalues = np.linalg.eigvalsh(matrix)
    clipped = np.clip(eigenvalues, EPS, None)
    logdet = float(np.sum(np.log(clipped)))
    trace_inv = float(np.sum(1.0 / clipped))
    min_eigenvalue = float(np.min(clipped))
    condition_number = float(np.max(clipped) / max(min_eigenvalue, EPS))
    return {
        "logdet": logdet,
        "trace_inv": trace_inv,
        "min_eigenvalue": min_eigenvalue,
        "condition_number": condition_number,
    }


def information_cost(
    FIM: Sequence[Sequence[float]],
    regularization: float = 1e-6,
    mode: str = "trace_inv",
) -> float:
    """Return an information cost for optimization or reward shaping.

    ``trace_inv`` is smaller for informative sensor geometries. ``neg_logdet``
    is also smaller when information volume is larger.
    """

    metrics = fim_metrics(FIM, regularization=regularization)
    normalized_mode = str(mode).lower()
    if normalized_mode in {"trace_inv", "a_optimal", "a-optimal"}:
        return float(metrics["trace_inv"])
    if normalized_mode in {"neg_logdet", "-logdet", "d_optimal", "d-optimal"}:
        return float(-metrics["logdet"])
    raise ValueError(f"Unknown information cost mode: {mode}")


def regularize_fim(FIM: Sequence[Sequence[float]], regularization: float = 1e-6) -> np.ndarray:
    """Return a symmetric regularized FIM."""

    matrix = np.asarray(FIM, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("FIM must have shape (3, 3).")
    matrix = 0.5 * (matrix + matrix.T)
    reg = max(float(regularization), 0.0)
    return matrix + reg * np.eye(3, dtype=float)


def _as_position(position: Sequence[float]) -> np.ndarray:
    array = np.asarray(position, dtype=float).reshape(-1)
    if array.size < 3:
        raise ValueError("Positions must contain at least three coordinates.")
    return array[:3].astype(float, copy=True)


def _as_covariance(covariance: Sequence[Sequence[float]] | Sequence[float] | float) -> np.ndarray:
    array = np.asarray(covariance, dtype=float)
    if array.ndim == 0:
        value = max(float(array), EPS)
        return np.eye(2, dtype=float) * value
    if array.ndim == 1:
        if array.size != 2:
            raise ValueError("A diagonal covariance vector must have length 2.")
        return np.diag(np.clip(array.astype(float), EPS, None))
    if array.shape != (2, 2):
        raise ValueError("Range-bearing covariance must be scalar, length-2, or 2x2.")
    matrix = 0.5 * (array + array.T)
    matrix += EPS * np.eye(2, dtype=float)
    return matrix
