"""Study 2 — vote prediction. For each respondent, the probability the model continues the backstory's
"In 2016, I voted for" with a Trump token versus a Clinton token (token sets of Appendix C.1, summed and
normalised), dichotomised at 0.50 and compared with the ANES self-reported vote by tetrachoric correlation,
Cohen's kappa and proportion agreement, overall and by the subgroups of the paper's Table 1."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from hsr.llm import LLM
from papers.out_of_one_many import data as D
from papers.out_of_one_many import prompts as P
from papers.out_of_one_many import stats as S


def token_probability(logprobs: dict[str, float]) -> Optional[float]:
    """P(Republican) after collapsing the token sets; None when neither set received probability mass."""
    pt = pc = 0.0
    for tok, lp in logprobs.items():
        t = tok.strip().lower()
        if t in P.TRUMP_TOKENS:
            pt += math.exp(lp)
        elif t in P.CLINTON_TOKENS:
            pc += math.exp(lp)
    if pt + pc == 0:
        return None
    return pt / (pt + pc)


def predict(llm: LLM, rows: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        row = r.to_dict()
        prompt = P.backstory(row)
        p = token_probability(llm.next_token_logprobs(prompt, k))
        out.append({"id": row["id"], "p_republican": p, "silicon_vote": (None if p is None else int(p > 0.5))})
    return pd.DataFrame(out)


SUBGROUPS: dict[str, Any] = {
    "Whole sample": lambda d: pd.Series(True, index=d.index),
    "Men": lambda d: d["gender"] == 1, "Women": lambda d: d["gender"] == 2,
    "Strong partisans": lambda d: d["pid7"].isin([1, 7]), "Weak partisans": lambda d: d["pid7"].isin([2, 6]),
    "Leaners": lambda d: d["pid7"].isin([3, 5]), "Independents": lambda d: d["pid7"] == 4,
    "Conservatives": lambda d: d["ideology"].isin([5, 6, 7]), "Moderates": lambda d: d["ideology"] == 4, "Liberals": lambda d: d["ideology"].isin([1, 2, 3]),
    "Whites": lambda d: d["race"] == 1, "Blacks": lambda d: d["race"] == 2, "Hispanics": lambda d: d["race"] == 5,
    "Attends church": lambda d: d["church"] == 1, "Doesn't attend church": lambda d: d["church"] == 2,
    "High interest in politics": lambda d: d["interest"].isin([1, 2]), "Low interest in politics": lambda d: d["interest"].isin([3, 4]),
    "Discusses politics": lambda d: d["discuss"] == 1, "Doesn't discuss politics": lambda d: d["discuss"] == 2,
    "18 to 30 years old": lambda d: d["age"] <= 30, "31 to 45 years old": lambda d: (d["age"] > 30) & (d["age"] <= 45),
    "46 to 60 years old": lambda d: (d["age"] > 45) & (d["age"] <= 60), "Over 60": lambda d: d["age"] > 60,
}


def evaluate(human: pd.DataFrame, silicon: pd.DataFrame) -> dict[str, Any]:
    """Two-party voters only (ANES vote 1 = Clinton -> 0, 2 = Trump -> 1), as in the paper's binary coding."""
    d = human.merge(silicon, on="id")
    d = d[d["vote"].isin([1, 2]) & d["silicon_vote"].notna()].copy()
    d["human_rep"] = (d["vote"] == 2).astype(int)
    d["silicon_rep"] = d["silicon_vote"].astype(int)
    # Threshold-free diagnostics (not in the paper): does P(Republican) rank voters correctly even when the model's
    # marginal is miscalibrated? AUC, Spearman, and a split at the quantile that matches the human Republican share.
    y, p = d["human_rep"].values, d["p_republican"].values.astype(float)
    diag: dict[str, Any] = {"mean_p_republican": float(p.mean()), "silicon_rep_share_at_0.5": float(d["silicon_rep"].mean()), "human_rep_share": float(y.mean())}
    if 0 < y.mean() < 1:
        diag["auc"] = float(sps.mannwhitneyu(p[y == 1], p[y == 0]).statistic / (np.sum(y == 1) * np.sum(y == 0)))
        diag["spearman"] = float(sps.spearmanr(p, y).statistic)
        thr = float(np.quantile(p, 1 - y.mean()))
        sil_q = (p > thr).astype(int)
        diag["quantile_matched"] = {"threshold": thr, "agreement": S.agreement(y, sil_q), "tetrachoric": S.tetrachoric(y, sil_q), "kappa": S.kappa(y, sil_q)}
    diag["mean_p_republican_by_pid7"] = {str(int(k)): float(v) for k, v in d.groupby("pid7")["p_republican"].mean().items()}
    diag["mean_p_republican_by_ideology"] = {str(int(k)): float(v) for k, v in d.groupby("ideology")["p_republican"].mean().items()}
    rows = {}
    for name, mask in SUBGROUPS.items():
        sub = d[mask(d).fillna(False)]
        if len(sub) < 5:
            rows[name] = {"n": int(len(sub))}
            continue
        rows[name] = {"n": int(len(sub)), "tetrachoric": S.tetrachoric(sub["human_rep"], sub["silicon_rep"]), "kappa": S.kappa(sub["human_rep"], sub["silicon_rep"]),
                      "agreement": S.agreement(sub["human_rep"], sub["silicon_rep"]), "human_rep_share": float(sub["human_rep"].mean()),
                      "silicon_rep_share": float(sub["silicon_rep"].mean()), "mean_p_republican": float(sub["p_republican"].mean())}
    return {"n_two_party": int(len(d)), "subgroups": rows, "diagnostics": diag}
