"""Connectivity helpers for aerial and underwater 3U links."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utils.physics import distance


MAX_EXPONENT = 700.0


@dataclass(frozen=True)
class ConnectivityMetrics:
    """End-to-end communication status for legacy experiment code."""

    uuv_to_usv_distances: np.ndarray
    uuv_connected: np.ndarray
    usv_uav_distance: float
    usv_uav_connected: bool

    @property
    def connected_fraction(self) -> float:
        if len(self.uuv_connected) == 0:
            return 0.0
        end_to_end = self.uuv_connected & self.usv_uav_connected
        return float(np.mean(end_to_end))

    @property
    def all_connected(self) -> bool:
        return bool(self.usv_uav_connected and np.all(self.uuv_connected))


def stable_exp(value: float, lower: float = -MAX_EXPONENT, upper: float = MAX_EXPONENT) -> float:
    """Exponentiate after clipping to avoid overflow and underflow surprises."""

    value = float(np.nan_to_num(float(value), nan=lower, neginf=lower, posinf=upper))
    return float(np.exp(np.clip(value, lower, upper)))


def uav_usv_connectivity_probability(
    uav_position: np.ndarray,
    usv_position: np.ndarray,
    T_a: float = 1.0,
    path_loss_exponent: float = 2.0,
    sigma2: float = 1e-9,
    interference_expectation: float = 0.0,
    mu: float = 1.0,
    p_a: float = 1.0,
) -> float:
    """Compute stable UAV-USV connectivity probability.

    Implements ``P_c = exp(-(T_a ||U-S||^a sigma2 + E[I_t]) / (mu p_a))``.
    The denominator is clamped to a small positive value and the exponent is
    clipped before calling ``exp`` so that extreme distances remain numerical.
    """

    link_distance = distance(uav_position, usv_position)
    denominator = max(float(mu) * float(p_a), 1e-300)
    numerator = float(T_a) * (link_distance ** float(path_loss_exponent)) * float(sigma2)
    numerator += max(float(interference_expectation), 0.0)
    exponent = -numerator / denominator
    return float(np.clip(stable_exp(exponent, lower=-MAX_EXPONENT, upper=0.0), 0.0, 1.0))


def underwater_adjacency_matrix(
    usv_position: np.ndarray,
    uuv_positions: np.ndarray,
    communication_range: float,
) -> np.ndarray:
    """Build an adjacency matrix for one USV plus ``M`` UUVs.

    Node 0 is the USV and nodes ``1..M`` are UUVs. An undirected edge is present
    when Euclidean distance is below ``communication_range``.
    """

    uuv_positions = np.asarray(uuv_positions, dtype=float)
    if uuv_positions.ndim == 1:
        uuv_positions = uuv_positions[None, :]
    nodes = np.vstack([np.asarray(usv_position, dtype=float), uuv_positions])
    n_nodes = nodes.shape[0]
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    communication_range = float(communication_range)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if distance(nodes[i], nodes[j]) <= communication_range:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    return adjacency


def soft_eigen_connectivity(adjacency: np.ndarray) -> Tuple[float, np.ndarray]:
    """Return ``delta_bar = log(mean(exp(delta_i)))`` for graph eigenvalues.

    The formula is evaluated with a log-sum-exp style stabilization. For an
    undirected adjacency matrix, eigenvalues are real; ``eigvalsh`` is used for
    numerical stability.
    """

    adjacency = np.asarray(adjacency, dtype=float)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    eigenvalues = np.linalg.eigvalsh(adjacency)
    max_eval = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    delta_bar = max_eval + float(np.log(np.mean(np.exp(eigenvalues - max_eval))))
    return delta_bar, eigenvalues


def underwater_connectivity_metric(
    usv_position: np.ndarray,
    uuv_positions: np.ndarray,
    communication_range: float,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute adjacency, eigenvalues, and soft eigenvalue connectivity."""

    adjacency = underwater_adjacency_matrix(usv_position, uuv_positions, communication_range)
    delta_bar, eigenvalues = soft_eigen_connectivity(adjacency)
    return delta_bar, adjacency, eigenvalues


def compute_connectivity(
    uuv_positions: np.ndarray,
    usv_position: np.ndarray,
    uav_position: np.ndarray,
    acoustic_range: float,
    radio_range: float,
) -> ConnectivityMetrics:
    """Compute simple threshold UUV-USV acoustic and USV-UAV radio links."""

    uuv_positions = np.asarray(uuv_positions, dtype=float)
    if uuv_positions.ndim == 1:
        uuv_positions = uuv_positions[None, :]
    usv_position = np.asarray(usv_position, dtype=float)
    uav_position = np.asarray(uav_position, dtype=float)

    uuv_distances = np.asarray([distance(uuv_position, usv_position) for uuv_position in uuv_positions], dtype=float)
    uuv_connected = uuv_distances <= float(acoustic_range)

    usv_uav_distance = distance(usv_position, uav_position)
    usv_uav_connected = usv_uav_distance <= float(radio_range)

    return ConnectivityMetrics(
        uuv_to_usv_distances=uuv_distances,
        uuv_connected=uuv_connected,
        usv_uav_distance=usv_uav_distance,
        usv_uav_connected=bool(usv_uav_connected),
    )
