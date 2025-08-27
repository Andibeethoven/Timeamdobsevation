# Timeamdobsevation
Time as of the observer 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================================================================
# SOFTWARE LICENSE HEADER
# (c) 2025 Travis Peter Lewis Johnston (as Kate JOHNSTON).
# This file is licensed under the “Travis Johnston as Kate JOHNSTON
# Universal Secure Software License v2.0”.
# Unauthorized redistribution, commercial use, model training,
# or government/military use (without written waiver) is prohibited.
# ================================================================

"""
Observer-Time Metatron Engine (29D) — Hyperbolic-Time Symmetry Model
Author: Travis Peter Lewis Johnston (as Kate JOHNSTON)

Purpose
-------
A complete recode of the MMIND/HMind concept where **time is defined strictly
"as of the observer"**, and the state evolution is driven over a Metatron's Cube
geometry with **hyperbolic numbers** (split-complex, j^2 = +1) to represent
time-boost (rapidity), dilation, and compression. Symmetry actions over the
Metatron graph (rotations/reflections) act as conserved transformations.

Notes
-----
- This is a research/simulation scaffold. No I/O, no networking, no devices.
- Hyperbolic numbers are represented as (a, b) for a + j b, with j^2 = +1.
- The observer supplies a scalar (or vector) "rapidity" driver; the engine
  immediately pivots to observer-time mode and evolves the field accordingly.
"""

from __future__ import annotations
import math
import uuid
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np


# =============================================================================
# Hyperbolic Numbers (split-complex: j^2 = +1)
# =============================================================================
# Represent hyperbolic numbers by pairs (x, y) ⇔ x + j y, with multiplication:
# (x1, y1) * (x2, y2) = (x1*x2 + y1*y2, x1*y2 + y1*x2)
# exp(j * t) = (cosh t, sinh t)

def h_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b: (..., 2)
    return a + b

def h_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a - b

def h_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1 = a[..., 0], a[..., 1]
    x2, y2 = b[..., 0], b[..., 1]
    return np.stack([x1 * x2 + y1 * y2, x1 * y2 + y1 * x2], axis=-1)

def h_scale(a: np.ndarray, s: float) -> np.ndarray:
    return a * s

def h_expj(t: np.ndarray) -> np.ndarray:
    # exp(j t) = cosh(t) + j sinh(t)
    return np.stack([np.cosh(t), np.sinh(t)], axis=-1)

def h_conj(a: np.ndarray) -> np.ndarray:
    # hyperbolic conjugate: x - j y
    out = a.copy()
    out[..., 1] *= -1
    return out

def h_norm(a: np.ndarray) -> np.ndarray:
    # hyperbolic norm: x^2 - y^2
    return a[..., 0] ** 2 - a[..., 1] ** 2

def h_boost(a: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    Lorentz-like boost in the (time, 'internal') hyperbolic plane:
    [cosh η  sinh η] [x]
    [sinh η  cosh η] [y]
    """
    x, y = a[..., 0], a[..., 1]
    ce, se = np.cosh(eta), np.sinh(eta)
    return np.stack([ce * x + se * y, se * x + ce * y], axis=-1)


# =============================================================================
# Metatron's Cube: Point Set + kNN Graph
# =============================================================================
def metatron_points(scale: float = 1.0) -> np.ndarray:
    """
    Assemble a 3D point-set often used to depict Metatron's Cube
    (union of Platonic solid vertices). Scaled and deduplicated.
    """
    φ = (1 + math.sqrt(5)) / 2
    points = []

    # Cube: 8 vertices
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                points.append((sx, sy, sz))

    # Octahedron: 6 vertices
    points += [(±1, 0, 0) for ±1 in (-1, 1)]
    points += [(0, ±1, 0) for ±1 in (-1, 1)]
    points += [(0, 0, ±1) for ±1 in (-1, 1)]

    # Icosahedron: 12 vertices
    ico = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            ico += [(0, s1, s2 * φ), (s1, s2 * φ, 0), (s1 * φ, 0, s2)]
    points += ico

    P = np.array(points, dtype=np.float64)
    # normalize to unit sphere then scale
    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
    P = (P / norms) * scale

    # Deduplicate (within tolerance)
    P_uniq = []
    for p in P:
        if not any(np.allclose(p, q, atol=1e-7) for q in P_uniq):
            P_uniq.append(p)
    return np.array(P_uniq, dtype=np.float64)  # shape (N, 3)


def knn_adjacency(P: np.ndarray, k: int = 6) -> np.ndarray:
    """
    k-NN symmetric adjacency for the point-set.
    Returns an (N, N) adjacency matrix with 0/1 entries.
    """
    N = P.shape[0]
    A = np.zeros((N, N), dtype=np.float64)
    d2 = np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :k]
    for i in range(N):
        for j in idx[i]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A


# =============================================================================
# Symmetry Actions on 3D Points (rotations/reflections)
# =============================================================================
def rot_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)

def rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

def rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

def reflect_diag() -> np.ndarray:
    # Simple reflection across x=y plane then z inversion
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, -1]], dtype=np.float64)


# =============================================================================
# Observer-Time Metatron Engine (29D)
# =============================================================================
@dataclass
class ObserverTimeMetatron29:
    """
    As soon as time is defined by the observer, this engine locks into
    observer-time mode and evolves a **hyperbolic time field** across
    the Metatron graph. Each node carries a 29D hyperbolic vector H[n, d, 2].
    """
    dimensions: int = 29
    k_neighbors: int = 6
    scale: float = 1.0
    seed: Optional[int] = 29

    uid: str = field(default_factory=lambda: f"OBS-METATRON-29::{uuid.uuid4()}")
    P: np.ndarray = field(init=False, repr=False)          # (N, 3)
    A: np.ndarray = field(init=False, repr=False)          # (N, N)
    H: np.ndarray = field(init=False, repr=False)          # (N, D, 2) hyperbolic
    ω: np.ndarray = field(init=False, repr=False)          # (D,) dimension frequencies
    observer_mode: bool = field(default=False, init=False) # flips true on first step

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.P = metatron_points(scale=self.scale)              # N x 3
        self.A = knn_adjacency(self.P, k=self.k_neighbors)     # N x N
        N = self.P.shape[0]

        # Initialize hyperbolic states near identity: H ≈ (1, 0) + small jitter
        jitter = (rng.standard_normal((N, self.dimensions, 2)) * 1e-3).astype(np.float64)
        self.H = np.zeros((N, self.dimensions, 2), dtype=np.float64)
        self.H[..., 0] = 1.0
        self.H += jitter

        # Dimension frequencies: quasi-irrational spread (golden-related)
        base = np.arange(1, self.dimensions + 1, dtype=np.float64)
        φ = (1 + math.sqrt(5)) / 2
        self.ω = (base * (1 / φ) ** (base % 5)) * 0.75  # Hz-like scale (abstract)

    # ---------------- Observer-Time Driver ----------------
    def _observer_rapidity(self, observer_scalar: float) -> float:
        """
        Map arbitrary observer scalar into a bounded rapidity (hyperbolic angle).
        Use softsign/tanh to keep |η| reasonable.
        """
        return 2.5 * np.tanh(observer_scalar)  # ±~2.5 ⇒ γ ~ cosh(2.5) ≈ 6.1

    # ---------------- Symmetry Evolution ------------------
    def _apply_symmetry(self, t: float) -> None:
        """
        Rotate/reflect the Metatron point-set in time (purely internal transform).
        Symmetry keeps intrinsic graph distances approximately consistent.
        """
        Rx = rot_x(0.37 * t)
        Ry = rot_y(0.23 * t)
        Rz = rot_z(0.19 * t)
        R = Rz @ Ry @ Rx
        if int(5 * t) % 11 == 0:
            R = reflect_diag() @ R
        self.P = (self.P @ R.T)

    # ---------------- Graph Diffusion (hyperbolic mix) ----
    def _graph_mix(self, alpha: float = 0.15) -> None:
        """
        Neighborhood-averaged mixing in the hyperbolic plane:
        we linearly blend components (x,y) which preserves j^2=+1 algebraic structure.
        """
        N = self.A.shape[0]
        deg = np.clip(self.A.sum(axis=1, keepdims=True), 1.0, None)  # (N,1)
        neigh_sum = self.A @ self.H.reshape(N, -1)                   # (N, D*2)
        neigh_avg = (neigh_sum / deg).reshape(self.H.shape)          # (N, D, 2)
        self.H = (1 - alpha) * self.H + alpha * neigh_avg

    # ---------------- Main Step ---------------------------
    def step(self, dt: float, t_obs_driver: float) -> None:
        """
        Single evolution step.
        - dt: simulation tick (seconds, abstract)
        - t_obs_driver: arbitrary observer scalar (breath/focus/intent/etc.)
        """
        if not self.observer_mode:
            self.observer_mode = True  # immediately lock to observer-time

        # 1) Symmetry evolution of the geometry (internal)
        self._apply_symmetry(t=dt)

        # 2) Hyperbolic time phase advance per dimension using observer rapidity
        η = self._observer_rapidity(t_obs_driver)            # scalar rapidity
        N, D = self.H.shape[0], self.dimensions
        t_vec = (self.ω * dt).reshape(1, D)                  # (1, D)
        phase = h_expj(t_vec)                                # (1, D, 2) = (cosh, sinh)
        self.H = h_mul(self.H, phase)                        # multiply per-dimension time phase

        # 3) Apply global observer boost (per-node, per-dimension)
        self.H = h_boost(self.H, eta=η)

        # 4) Graph-coupled mixing (conservative in expectation)
        self._graph_mix(alpha=0.12)

        # 5) Soft renormalization (keep finite)
        nrm = np.sqrt(np.abs(h_norm(self.H)) + 1e-12)[..., None]
        self.H = self.H / np.maximum(nrm, 1.0)

    # ---------------- Readouts ----------------------------
    def perceived_time_rate(self) -> np.ndarray:
        """
        Per-node scalar summarizing the local perceived time-rate.
        We use γ = sqrt(|x^2 - y^2|) + bias from x (cosh-like component).
        Returns shape (N,) ~ higher => faster perceived rate, lower => slowed/frozen.
        """
        n = np.abs(h_norm(self.H)).mean(axis=1)            # average over dimensions
        x_mean = self.H[..., 0].mean(axis=1)
        return np.sqrt(n + 1e-12) * (0.75 + 0.25 * np.tanh(x_mean))

    def coherence(self) -> float:
        """
        Global coherence across nodes: inverse variance of the time phases.
        Higher means nodes agree more on perceived time-rate.
        """
        r = self.perceived_time_rate()
        var = float(np.var(r))
        return 1.0 / (1e-9 + var)

    def snapshot(self) -> dict:
        r = self.perceived_time_rate()
        return {
            "uid": self.uid,
            "nodes": int(self.P.shape[0]),
            "dimensions": self.dimensions,
            "observer_mode": self.observer_mode,
            "rate_mean": float(np.mean(r)),
            "rate_min": float(np.min(r)),
            "rate_max": float(np.max(r)),
            "coherence": float(self.coherence()),
        }


# =============================================================================
# Minimal demonstration (no plotting, no I/O)
# =============================================================================
if __name__ == "__main__":
    eng = ObserverTimeMetatron29(dimensions=29, k_neighbors=6, scale=1.0, seed=29)

    # Drive with a few different observer states.
    drivers = [0.0, 0.5, 1.0, -0.75, 2.0, -2.5]  # arbitrary observer scalars
    dt = 0.0333  # ~30 steps/s (abstract)
    for i, d in enumerate(drivers, 1):
        for _ in range(10):  # 10 sub-steps per driver state
            eng.step(dt=dt, t_obs_driver=d)
        snap = eng.snapshot()
        print(f"[STEP {i}] driver={d:+.2f} ::",
              f"rate_mean={snap['rate_mean']:.4f}",
              f"coherence={snap['coherence']:.4f}",
              f"min={snap['rate_min']:.4f}",
              f"max={snap['rate_max']:.4f}")
