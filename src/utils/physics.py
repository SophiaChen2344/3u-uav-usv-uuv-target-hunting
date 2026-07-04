"""Numerical physics helpers for the 3U reproduction.

The simulator uses meters, seconds, Newtons, Joules, and m/s unless a function
explicitly says otherwise. Depth in the simplified environment is represented
with negative z values, while a few compatibility wrappers still support older
positive-depth helpers.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


EPS = 1e-12
KNOT_TO_MPS = 0.514444


def safe_norm(vector: np.ndarray, eps: float = EPS) -> float:
    """Return a finite Euclidean norm with tiny values rounded to zero."""

    value = float(np.linalg.norm(np.asarray(vector, dtype=float)))
    if not np.isfinite(value) or value < eps:
        return 0.0
    return value


def safe_divide(numerator: float | np.ndarray, denominator: float, eps: float = EPS) -> float | np.ndarray:
    """Divide by a scalar while avoiding zero and non-finite denominators."""

    denominator = float(denominator)
    if not np.isfinite(denominator) or abs(denominator) < eps:
        denominator = eps
    return numerator / denominator


def safe_unit_vector(vector: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return ``vector / ||vector||`` or a zero vector when the norm is tiny."""

    vector = np.asarray(vector, dtype=float)
    norm = safe_norm(vector, eps=eps)
    if norm <= 0.0:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two points."""

    return safe_norm(np.asarray(b, dtype=float) - np.asarray(a, dtype=float))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compatibility alias for :func:`distance`."""

    return distance(a, b)


def unit_vector(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return the unit vector pointing from point ``a`` to point ``b``."""

    return safe_unit_vector(np.asarray(b, dtype=float) - np.asarray(a, dtype=float), eps=eps)


def clamp_position_to_region(
    position: np.ndarray,
    area_size: float = 400.0,
    z_min: float | None = -120.0,
    z_max: float | None = None,
) -> np.ndarray:
    """Clamp a position into the square operating region.

    ``x`` and ``y`` are clamped to ``[0, area_size]``. The z coordinate is
    clamped only when ``z_min`` or ``z_max`` are provided. This makes the helper
    usable for UAVs above the surface, USVs at ``z=0``, and UUVs/targets at
    negative depths.
    """

    clipped = np.asarray(position, dtype=float).copy()
    clipped[0] = np.clip(clipped[0], 0.0, float(area_size))
    clipped[1] = np.clip(clipped[1], 0.0, float(area_size))
    if clipped.size >= 3:
        if z_min is not None:
            clipped[2] = max(float(z_min), clipped[2])
        if z_max is not None:
            clipped[2] = min(float(z_max), clipped[2])
    return clipped


def knots_to_mps(knots: float | np.ndarray) -> float | np.ndarray:
    """Convert speed from nautical miles per hour to meters per second."""

    converted = np.asarray(knots, dtype=float) * KNOT_TO_MPS
    if np.isscalar(knots):
        return float(converted)
    return converted


def clip_position(position: np.ndarray, area_size: float, max_depth: float) -> np.ndarray:
    """Legacy positive-depth clipping helper.

    Older code in this repository used ``z=0`` at the surface and positive
    depth downward. Newer environment code uses negative depth, so new code
    should prefer :func:`clamp_position_to_region`.
    """

    clipped = np.asarray(position, dtype=float).copy()
    clipped[0] = np.clip(clipped[0], 0.0, float(area_size))
    clipped[1] = np.clip(clipped[1], 0.0, float(area_size))
    clipped[2] = np.clip(clipped[2], 0.0, float(max_depth))
    return clipped


def move_toward(position: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    """Move from ``position`` toward ``target`` by at most ``max_step`` meters."""

    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    delta = target - position
    norm = safe_norm(delta)
    if norm <= EPS or norm <= max_step:
        return target.copy()
    return position + delta / norm * float(max_step)


def camera_footprint_radius(height: float, fov_deg: float, min_radius: float) -> float:
    """Approximate a nadir camera ground footprint radius from altitude and FOV."""

    half_angle = math.radians(float(fov_deg)) / 2.0
    return max(float(min_radius), float(height) * math.tan(half_angle))


def random_unit_vector_2d(rng: np.random.Generator) -> np.ndarray:
    """Sample a random 2D unit vector."""

    theta = rng.uniform(0.0, 2.0 * math.pi)
    return np.array([math.cos(theta), math.sin(theta)], dtype=float)


def linearly_normalize(value: float, scale: float) -> float:
    """Normalize a scalar and keep extreme observations bounded."""

    if scale <= 0:
        return 0.0
    return float(np.clip(float(value) / float(scale), -2.0, 2.0))


def heading_to_action_delta(action: int) -> Tuple[float, float, float]:
    """Map a legacy primitive UUV action id to a unit displacement direction."""

    primitive_moves = {
        0: (0.0, 0.0, 0.0),
        1: (0.0, 1.0, 0.0),
        2: (0.0, -1.0, 0.0),
        3: (1.0, 0.0, 0.0),
        4: (-1.0, 0.0, 0.0),
        5: (0.0, 0.0, -1.0),
        6: (0.0, 0.0, 1.0),
    }
    return primitive_moves[int(action)]
