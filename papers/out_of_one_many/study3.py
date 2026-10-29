"""Study 3 — closed-ended questions. For each respondent and each of the twelve items as the target, the
interview template with the other items answered is completed by the model (5 tokens, temperature 0.7, as in
the paper), the completion is lower-cased, stripped and matched exactly to the allowed options (else missing),
and Cramer's V is computed for every pair of items in the human data and between the human conditioning values
and the generated answers."""

from __future__ import annotations

import itertools
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from hsr.llm import LLM
from papers.out_of_one_many import data as D
from papers.out_of_one_many import prompts as P
from papers.out_of_one_many import stats as S

PAIR_VARS = ["gender", "race", "age", "education", "church", "patriotism", "discuss", "interest", "ideology", "pid7", "vote2016"]
AGE_BINS = [(18, 30), (31, 45), (46, 60), (61, 200)]


def parse_answer(item: str, text: str) -> Optional[int]:
    """Exact match against the allowed responses after lower-casing and stripping (Appendix D.1); the completion
    is cut at the first newline so a model that continues the interview is read only for its first answer."""
    t = text.split("\n")[0].strip().lower().strip(".\"' ")
    if item == "age":
        m = re.match(r"(\d{1,3})", t)
        return int(m.group(1)) if m and 18 <= int(m.group(1)) <= 100 else None
    opts = {v.lower(): k for k, v in P.OPTIONS[item].items()}
    if t in opts:
        return opts[t]
    # the longest option that the first line starts with (e.g. "strong republican, because ...")
    for text_opt, code in sorted(opts.items(), key=lambda kv: -len(kv[0])):
        if t.startswith(text_opt):
            return code
    return None


def generate(llm: LLM, rows: pd.DataFrame, max_tokens: int = 5, temperature: float = 0.7, seed: int = 0) -> pd.DataFrame:
    out = []
    for _, r in rows.iterrows():
        row = r.to_dict()
        rec = {"id": row["id"]}
        for target in D.ITEMS:
            if target == "vote" and row.get("voted") == 0:
                rec[target] = None  # not asked of non-voters
                continue
            text = llm.complete_raw(P.interview(row, target), max_tokens=max_tokens, temperature=temperature, seed=seed)
            rec[target] = parse_answer(target, text)
            rec[target + "_raw"] = text
        out.append(rec)
    return pd.DataFrame(out)


def age_binned(age: pd.Series) -> pd.Series:
    def f(a):
        if pd.isna(a):
            return np.nan
        for i, (lo, hi) in enumerate(AGE_BINS):
            if lo <= a <= hi:
                return i
        return np.nan
    return age.map(f)


def analysis_frame(h: pd.DataFrame, s: Optional[pd.DataFrame] = None, age_as: str = "raw") -> pd.DataFrame:
    """Items in the form used for Cramer's V. Age is kept as the raw integer (a categorical with ~70 levels): with
    that coding our human Cramer's V reproduces the paper's appendix values (r = 0.999); with four age bands the
    age pairs come out far lower (e.g. age x gender 0.05 vs the paper's 0.18). vote2016 = turnout and choice combined."""
    f = pd.DataFrame({"id": h["id"]})
    src = h if s is None else s
    for v in PAIR_VARS:
        if v == "age":
            f[v] = src["age"].values if age_as == "raw" else age_binned(src["age"]).values
        elif v == "vote2016":
            f[v] = D.vote_combined(src["voted"], src["vote"]).values
        else:
            f[v] = src[v].values
    return f


def cramers_matrix(h_frame: pd.DataFrame, s_frame: Optional[pd.DataFrame] = None, complete_only: bool = True) -> dict[str, Any]:
    """Human: V(x_h, y_h). Silicon (paper's definition): V(x_h, y_s) — the ANES conditioning value of x against the
    generated answer for y. Returns per-pair values plus the mean difference and correlation of the 55 pairs."""
    if s_frame is None:
        s_frame = h_frame
    cols = PAIR_VARS
    if complete_only:
        keep = h_frame[cols].notna().all(axis=1) & s_frame[cols].notna().all(axis=1)
        h_frame, s_frame = h_frame[keep], s_frame[keep]
    pairs = {}
    for x, y in itertools.combinations(cols, 2):
        hv = S.cramers_v(h_frame[x], h_frame[y])
        sv = S.cramers_v(h_frame[x], s_frame[y])
        pairs[f"{x}|{y}"] = {"human": hv, "silicon": sv, "diff": hv - sv}
    diffs = np.array([p["diff"] for p in pairs.values()])
    hs = np.array([p["human"] for p in pairs.values()]); ss = np.array([p["silicon"] for p in pairs.values()])
    ok = ~(np.isnan(hs) | np.isnan(ss))  # a pair is undefined when a variable has a single category in the (silicon) sample
    return {"n_complete": int(len(h_frame)), "pairs": pairs, "n_pairs_defined": int(ok.sum()), "mean_diff_human_minus_silicon": float(np.nanmean(diffs)),
            "mean_abs_diff": float(np.nanmean(np.abs(diffs))), "correlation": float(np.corrcoef(hs[ok], ss[ok])[0, 1]) if ok.sum() > 2 else float("nan")}
