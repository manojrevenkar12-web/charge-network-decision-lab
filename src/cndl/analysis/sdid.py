"""Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens and Wager, 2021).

Control sites are weighted so their pre-period path matches the treated sites' path, and pre
periods are weighted so they resemble the post period; the effect is a weighted double
difference. When treatment was given to sites that were growing faster, the unit weights favour
controls that grew the same way, which a plain difference-in-differences cannot do.

Staggered adoption is handled cohort by cohort (sites first treated in the same week) against
never-treated sites, and the cohort effects are averaged with weights equal to their number of
treated site-weeks, so the estimand matches the average effect over treated site-weeks.
"""

from __future__ import annotations

import numpy as np


def _fw_simplex(A: np.ndarray, b: np.ndarray, zeta: float, iters: int = 600, intercept: bool = True) -> np.ndarray:
    """min_w ||A w + c - b||^2 + zeta^2 * n_rows * ||w||^2 over the simplex, c a free intercept.

    Frank-Wolfe with exact line search, as in the reference implementation."""
    if intercept:
        A = A - A.mean(axis=0, keepdims=True)
        b = b - b.mean()
    n, k = A.shape
    eta = n * zeta**2
    w = np.full(k, 1.0 / k)
    Aw = A @ w
    for _ in range(iters):
        grad = 2 * (A.T @ (Aw - b)) + 2 * eta * w
        j = int(np.argmin(grad))
        d = -w.copy()
        d[j] += 1.0
        Ad = A[:, j] - Aw
        denom = Ad @ Ad + eta * (d @ d)
        if denom <= 1e-12:
            break
        step = -((Aw - b) @ Ad + eta * (w @ d)) / denom
        step = float(np.clip(step, 0.0, 1.0))
        if step < 1e-10:
            break
        w = w + step * d
        Aw = Aw + step * Ad
    return w


def sdid_block(Y_co: np.ndarray, Y_tr: np.ndarray, n_pre: int) -> float:
    """Y_co: controls x periods, Y_tr: treated x periods (balanced); first n_pre periods are pre."""
    pre_co, post_co = Y_co[:, :n_pre], Y_co[:, n_pre:]
    pre_tr, post_tr = Y_tr[:, :n_pre], Y_tr[:, n_pre:]
    n_tr = Y_tr.shape[0]
    t_post = Y_co.shape[1] - n_pre
    # regularisation from the paper: zeta = (N_tr * T_post)^(1/4) * sd of first differences of controls
    diffs = np.diff(pre_co, axis=1)
    zeta = (n_tr * t_post) ** 0.25 * float(np.std(diffs))
    omega = _fw_simplex(pre_co.T, pre_tr.mean(axis=0), zeta)
    lam = _fw_simplex(pre_co, post_co.mean(axis=1), 1e-6 * float(np.std(diffs)) + 1e-12)
    tr_post = post_tr.mean()
    tr_pre = pre_tr.mean(axis=0) @ lam
    co_post = omega @ post_co.mean(axis=1)
    co_pre = omega @ (pre_co @ lam)
    return float((tr_post - tr_pre) - (co_post - co_pre))


def staggered(Y: np.ndarray, first: np.ndarray, min_pre: int = 20) -> tuple[float, dict[int, tuple[float, int]]]:
    """Y: sites x periods (balanced, no NaN); first: first treated period per site, -1 if never.

    Returns the weighted average effect and per-cohort (effect, treated site-periods)."""
    never = first < 0
    cohorts = sorted(set(first[first >= 0].tolist()))
    out: dict[int, tuple[float, int]] = {}
    for g in cohorts:
        if g < min_pre:
            continue
        tr = first == g
        # compare only up to the week the next cohort is treated? not needed: controls are never-treated
        est = sdid_block(Y[never], Y[tr], g)
        out[g] = (est, int(tr.sum() * (Y.shape[1] - g)))
    total = sum(n for _, n in out.values())
    att = sum(e * n for e, n in out.values()) / total
    return float(att), out


def bootstrap(Y: np.ndarray, first: np.ndarray, reps: int, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        if (first[idx] < 0).sum() < 10 or (first[idx] >= 0).sum() < 10:
            continue
        out.append(staggered(Y[idx], first[idx])[0])
    return np.array(out)
