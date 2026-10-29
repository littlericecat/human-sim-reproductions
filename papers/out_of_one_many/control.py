"""Distribution-level control (Bisbee, Clinton, Dorff, Kenkel & Larson, Political Analysis 2024, doi:10.1017/pan.2024.5):
silicon samples can match human means while being too homogeneous and while regression coefficients diverge.
For every Study 3 item we compare the full human and silicon response distributions (total-variation distance,
Shannon entropy, modal share), the within-partisan-group distributions, and for the vote the coefficients of a
logistic regression of Trump-vs-Clinton on ideology and party identification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from papers.out_of_one_many import data as D
from papers.out_of_one_many import prompts as P
from papers.out_of_one_many import stats as S

ORDINAL = ("patriotism", "interest", "ideology", "pid7", "education")


def item_distributions(h: pd.DataFrame, s: pd.DataFrame, items: list[str] = D.ITEMS) -> dict[str, Any]:
    out = {}
    for item in items:
        if item == "age":
            cats = list(range(18, 101))
        else:
            cats = sorted(P.OPTIONS[item].keys())
        keep = h[item].notna() & s[item].notna()
        hv, sv = h.loc[keep, item].astype(int), s.loc[keep, item].astype(int)
        if len(hv) == 0:
            continue
        ph, ps = S.distribution(hv, cats), S.distribution(sv, cats)
        rec = {"n": int(len(hv)), "tv_distance": S.total_variation(ph, ps), "entropy_human": S.entropy(ph), "entropy_silicon": S.entropy(ps),
               "modal_share_human": float(ph.max()), "modal_share_silicon": float(ps.max()), "mean_human": float(hv.mean()), "mean_silicon": float(sv.mean())}
        if item in ORDINAL or item == "age":
            rec["sd_human"], rec["sd_silicon"] = float(hv.std(ddof=1)), float(sv.std(ddof=1))
        if item != "age":
            rec["human_dist"] = {str(c): round(float(p), 4) for c, p in zip(cats, ph)}
            rec["silicon_dist"] = {str(c): round(float(p), 4) for c, p in zip(cats, ps)}
        out[item] = rec
    return out


def within_group(h: pd.DataFrame, s: pd.DataFrame, item: str, group_col: str = "pid7") -> dict[str, Any]:
    """Human vs silicon distribution of `item` inside each level of the human grouping variable."""
    cats = sorted(P.OPTIONS[item].keys())
    out = {}
    for g in sorted(h[group_col].dropna().unique()):
        keep = (h[group_col] == g) & h[item].notna() & s[item].notna()
        if keep.sum() < 10:
            continue
        ph, ps = S.distribution(h.loc[keep, item].astype(int), cats), S.distribution(s.loc[keep, item].astype(int), cats)
        out[str(int(g))] = {"n": int(keep.sum()), "tv_distance": S.total_variation(ph, ps), "entropy_human": S.entropy(ph), "entropy_silicon": S.entropy(ps)}
    return out


def vote_regression(h: pd.DataFrame, s_vote: pd.Series) -> dict[str, Any]:
    """Logit(Trump | ideology, pid7) fitted on the human vote and on the silicon vote with the same (human) covariates."""
    keep = h["vote"].isin([1, 2]) & s_vote.isin([1, 2]) & h["ideology"].notna() & h["pid7"].notna()
    X = np.column_stack([h.loc[keep, "ideology"].astype(float) - 4, h.loc[keep, "pid7"].astype(float) - 4])
    yh = (h.loc[keep, "vote"] == 2).astype(float).values
    ys = (s_vote[keep] == 2).astype(float).values
    wh, ws = S.logistic_fit(X, yh), S.logistic_fit(X, ys)
    names = ["intercept", "ideology (centered)", "pid7 (centered)"]
    return {"n": int(keep.sum()), "human": dict(zip(names, map(float, wh))), "silicon": dict(zip(names, map(float, ws)))}
