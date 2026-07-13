"""Energy helpers for UUV motion and underwater acoustic communication.

The formulas are approximate, paper-shaped models intended for reproducible
simulation rather than calibrated hardware prediction. Distances are meters
unless stated otherwise; acoustic communication distance ``l`` is converted to
kilometers for the attenuation term by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from utils.physics import safe_norm


MAX_EXPONENT = 60.0


@dataclass(frozen=True)
class UuvEnergyBreakdown:
    """Motion, communication, and total energy in Joules."""

    motion: float
    communication: float
    total: float


def acoustic_attenuation_gamma(frequency_khz: float) -> float:
    """Return the acoustic attenuation multiplier ``gamma(f)``.

    The paper gives a Thorp-like relation:

    ``10 log10 gamma = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2)
    + 2.75e-4 f^2 + 0.003``.

    Here ``f`` is interpreted in kHz and the right-hand side is treated as a
    dB/km absorption value. The returned multiplier is ``10 ** (alpha / 10)``.
    """

    f = max(float(frequency_khz), 0.0)
    f2 = f * f
    alpha_db = 0.11 * f2 / (1.0 + f2)
    alpha_db += 44.0 * f2 / (4100.0 + f2)
    alpha_db += 2.75e-4 * f2
    alpha_db += 0.003
    return float(10.0 ** (alpha_db / 10.0))


def motion_energy(
    velocity: np.ndarray | float,
    travel_time: float,
    epsilon: float = 0.8,
    drag_force: float = 2000.0,
) -> float:
    """Compute UUV motion energy ``E_m = t_h * epsilon * F_d * ||V_G||``."""

    speed = safe_norm(np.asarray(velocity, dtype=float))
    travel_time = max(float(travel_time), 0.0)
    propulsion_power = float(epsilon) * float(drag_force) * speed
    return float(travel_time * propulsion_power)


def communication_energy(
    bits: float,
    distance_m: float,
    frequency_khz: float = 10.0,
    circuit_energy_per_bit: float = 50e-9,
    q: float = 1e-12,
    bit_duration: float = 1e-3,
    distance_loss_factor: float | None = None,
    spreading_factor: float = 1.5,
    distance_in_km: bool = False,
) -> float:
    """Compute acoustic communication energy ``E_c = E_t + E_r``.

    ``E_t(k,l) = k E_u + q k T_b d_l exp(gamma(f) l)`` and
    ``E_r(k,l) = k E_u``. If ``distance_loss_factor`` is not supplied, the
    distance loss term ``d_l`` is approximated as
    ``max(l, eps) ** spreading_factor``. The exponential argument is clipped
    for numerical stability because the literal formula grows very quickly.
    """

    k = max(float(bits), 0.0)
    if k == 0.0:
        return 0.0

    l = max(float(distance_m), 0.0)
    l_for_attenuation = l if distance_in_km else l / 1000.0
    gamma = acoustic_attenuation_gamma(frequency_khz)
    exponent = float(np.clip(gamma * l_for_attenuation, 0.0, MAX_EXPONENT))
    if distance_loss_factor is None:
        distance_loss = max(l_for_attenuation, 1e-12) ** float(spreading_factor)
    else:
        distance_loss = max(float(distance_loss_factor), 0.0)

    transmit = k * float(circuit_energy_per_bit)
    transmit += float(q) * k * float(bit_duration) * distance_loss * float(np.exp(exponent))
    receive = k * float(circuit_energy_per_bit)
    return float(transmit + receive)


def total_uuv_energy(
    velocity: np.ndarray | float,
    travel_time: float,
    bits: float = 0.0,
    communication_distance_m: float = 0.0,
    motion_only: bool = False,
    energy_config: Mapping[str, float | bool] | None = None,
) -> UuvEnergyBreakdown:
    """Return ``E_UUV = E_m + E_c`` with an optional motion-only switch.

    ``motion_only`` mirrors the paper's evaluation note that communication
    energy is often much smaller than propulsion energy. Values in
    ``energy_config`` override the defaults by key.
    """

    cfg = dict(energy_config or {})
    use_motion_only = bool(cfg.get("motion_only", motion_only))
    e_motion = motion_energy(
        velocity=velocity,
        travel_time=travel_time,
        epsilon=float(cfg.get("epsilon", 0.8)),
        drag_force=float(cfg.get("drag_force", cfg.get("F_d", 2000.0))),
    )
    e_comm = 0.0
    if not use_motion_only:
        e_comm = communication_energy(
            bits=float(cfg.get("bits", bits)),
            distance_m=float(cfg.get("communication_distance_m", communication_distance_m)),
            frequency_khz=float(cfg.get("frequency_khz", 10.0)),
            circuit_energy_per_bit=float(cfg.get("circuit_energy_per_bit", cfg.get("E_u", 50e-9))),
            q=float(cfg.get("q", 1e-12)),
            bit_duration=float(cfg.get("bit_duration", cfg.get("T_b", 1e-3))),
            distance_loss_factor=cfg.get("distance_loss_factor", cfg.get("d_l")),
            spreading_factor=float(cfg.get("spreading_factor", 1.5)),
            distance_in_km=bool(cfg.get("distance_in_km", False)),
        )
    return UuvEnergyBreakdown(motion=e_motion, communication=e_comm, total=e_motion + e_comm)


def uuv_group_step_energy(
    displacement: np.ndarray | float,
    dt: float,
    num_uuvs: int = 1,
    communication_distance_m: float = 0.0,
    connected: bool = True,
    energy_config: Mapping[str, float | bool | str] | None = None,
) -> UuvEnergyBreakdown:
    """Return one-step energy for a formation-center UUV action.

    Both the simulator and Flow Matching scoring call this helper so generated
    trajectories and executed steps use the same energy model. The default
    ``quadratic`` mode preserves the compact simulator's historical scale; set
    ``model: physical`` to use the propulsion/acoustic formulas above.
    """

    cfg = dict(energy_config or {})
    model = str(cfg.get("model", "quadratic")).lower()
    n_uuvs = max(int(num_uuvs), 1)
    dt = max(float(dt), 1e-12)
    distance = safe_norm(np.asarray(displacement, dtype=float))
    speed = distance / dt

    if model in {"physical", "paper"}:
        bits = float(cfg.get("bits_per_step", cfg.get("bits", 0.0)))
        if not bool(connected):
            bits = 0.0
        per_uuv = total_uuv_energy(
            velocity=speed,
            travel_time=dt,
            bits=bits,
            communication_distance_m=float(communication_distance_m),
            motion_only=bool(cfg.get("motion_only", False)),
            energy_config=cfg,
        )
        return UuvEnergyBreakdown(
            motion=float(n_uuvs * per_uuv.motion),
            communication=float(n_uuvs * per_uuv.communication),
            total=float(n_uuvs * per_uuv.total),
        )

    base = float(cfg.get("energy_base", cfg.get("base", 2.0))) * dt
    linear = float(cfg.get("energy_linear", cfg.get("linear", 0.8))) * distance
    quadratic = float(cfg.get("energy_quadratic", cfg.get("quadratic", 0.04))) * speed**2
    motion = float(n_uuvs * (base + linear + quadratic))
    communication = acoustic_comm_energy(
        distance=float(communication_distance_m),
        connected=bool(connected),
        dt=dt,
        bits_per_second=float(cfg.get("bits_per_second", 0.0)),
        frequency_khz=float(cfg.get("frequency_khz", 10.0)),
        circuit_energy_per_bit=float(cfg.get("circuit_energy_per_bit", cfg.get("E_u", 50e-9))),
        q=float(cfg.get("q", 1e-12)),
        bit_duration=float(cfg.get("bit_duration", cfg.get("T_b", 1e-3))),
        spreading_factor=float(cfg.get("spreading_factor", 1.5)),
        distance_in_km=bool(cfg.get("distance_in_km", False)),
    )
    communication *= n_uuvs
    if bool(cfg.get("motion_only", True)):
        communication = 0.0
    return UuvEnergyBreakdown(motion=motion, communication=float(communication), total=float(motion + communication))


def uuv_motion_energy(
    displacement: np.ndarray,
    dt: float,
    epsilon: float = 0.8,
    drag_force: float = 2000.0,
    **_: float,
) -> float:
    """Compatibility wrapper for motion energy from displacement over ``dt``."""

    dt = max(float(dt), 1e-12)
    velocity = np.asarray(displacement, dtype=float) / dt
    return motion_energy(velocity=velocity, travel_time=dt, epsilon=epsilon, drag_force=drag_force)


def acoustic_comm_energy(
    distance: float,
    connected: bool = True,
    dt: float = 1.0,
    bits_per_second: float = 128.0,
    **kwargs: float,
) -> float:
    """Compatibility wrapper for per-step acoustic communication energy."""

    if not connected:
        return 0.0
    return communication_energy(bits=float(bits_per_second) * max(float(dt), 0.0), distance_m=distance, **kwargs)
