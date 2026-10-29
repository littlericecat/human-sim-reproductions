"""Statistics used by the paper and by the control: tetrachoric correlation (bivariate-normal MLE on a 2x2 table),
Cohen's kappa, proportion agreement, Cramer's V (bias-uncorrected, as in R's rcompanion/lsr default used by most
replication code), total-variation distance and entropy."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy import optimize, stats


def table2x2(x: Sequence[int], y: Sequence[int]) -> np.ndarray:
    x, y = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    t = np.zeros((2, 2), dtype=float)
    for a in (0, 1):
        for b in (0, 1):
            t[a, b] = np.sum((x == a) & (y == b))
    return t


def tetrachoric(x: Sequence[int], y: Sequence[int]) -> float:
    """MLE of the latent correlation of a bivariate normal dichotomised at the observed marginals (what R's
    psych::tetrachoric estimates). A cell count of 0 is replaced by 0.5 (psych's default correction)."""
    t = table2x2(x, y)
    if t.sum() == 0:
        return float("nan")
    t = np.where(t == 0, 0.5, t)
    n = t.sum()
    p_x1, p_y1 = t[1].sum() / n, t[:, 1].sum() / n
    if p_x1 in (0, 1) or p_y1 in (0, 1):
        return float("nan")
    h, k = stats.norm.ppf(1 - p_x1), stats.norm.ppf(1 - p_y1)  # thresholds

    def cell_probs(r: float) -> np.ndarray:
        bv = stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]])
        p11 = bv.cdf([-h, -k]) if False else None  # placeholder for readability
        # P(X=1,Y=1) = P(Z1>h, Z2>k)
        p_both = 1 - stats.norm.cdf(h) - stats.norm.cdf(k) + bv.cdf([h, k])
        p_x = 1 - stats.norm.cdf(h)
        p_y = 1 - stats.norm.cdf(k)
        p10 = p_x - p_both
        p01 = p_y - p_both
        p00 = 1 - p_x - p_y + p_both
        return np.clip(np.array([[p00, p01], [p10, p_both]]), 1e-12, 1)

    def nll(r: float) -> float:
        return -float(np.sum(t * np.log(cell_probs(r))))

    res = optimize.minimize_scalar(nll, bounds=(-0.999, 0.999), method="bounded")
    return float(res.x)


def kappa(x: Sequence[int], y: Sequence[int]) -> float:
    t = table2x2(x, y)
    n = t.sum()
    if n == 0:
        return float("nan")
    po = np.trace(t) / n
    pe = (t[0].sum() * t[:, 0].sum() + t[1].sum() * t[:, 1].sum()) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def agreement(x: Sequence[int], y: Sequence[int]) -> float:
    x, y = np.asarray(x), np.asarray(y)
    return float(np.mean(x == y)) if len(x) else float("nan")


def cramers_v(x: Sequence, y: Sequence) -> float:
    x, y = np.asarray(x), np.asarray(y)
    cats_x, cats_y = sorted(set(x.tolist())), sorted(set(y.tolist()))
    if len(cats_x) < 2 or len(cats_y) < 2:
        return float("nan")
    table = np.zeros((len(cats_x), len(cats_y)))
    ix, iy = {c: i for i, c in enumerate(cats_x)}, {c: i for i, c in enumerate(cats_y)}
    for a, b in zip(x, y):
        table[ix[a], iy[b]] += 1
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(math.sqrt(chi2 / (n * (min(table.shape) - 1))))


def distribution(values: Sequence, categories: Sequence) -> np.ndarray:
    v = np.asarray(values)
    counts = np.array([np.sum(v == c) for c in categories], dtype=float)
    return counts / counts.sum() if counts.sum() else counts


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-4, iters: int = 100) -> np.ndarray:
    """Newton–Raphson logistic regression with an intercept and a tiny ridge for separability; returns coefficients."""
    X = np.column_stack([np.ones(len(X)), X]).astype(float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w
