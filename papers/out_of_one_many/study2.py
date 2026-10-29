"""Study 2 — vote prediction. For each respondent, the probability the model continues the backstory's
"In 2016, I voted for" with a Trump token versus a Clinton token (token sets of Appendix C.1, summed and
normalised), dichotomised at 0.50 and compared with the ANES self-reported vote by tetrachoric correlation,
Cohen's kappa and proportion agreement, overall and by the subgroups of the paper's Table 1."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

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
    rows = {}
    for name, mask in SUBGROUPS.items():
        sub = d[mask(d).fillna(False)]
        if len(sub) < 5:
            rows[name] = {"n": int(len(sub))}
            continue
        rows[name] = {"n": int(len(sub)), "tetrachoric": S.tetrachoric(sub["human_rep"], sub["silicon_rep"]), "kappa": S.kappa(sub["human_rep"], sub["silicon_rep"]),
                      "agreement": S.agreement(sub["human_rep"], sub["silicon_rep"]), "human_rep_share": float(sub["human_rep"].mean()),
                      "silicon_rep_share": float(sub["silicon_rep"].mean()), "mean_p_republican": float(sub["p_republican"].mean())}
    return {"n_two_party": int(len(d)), "subgroups": rows}
