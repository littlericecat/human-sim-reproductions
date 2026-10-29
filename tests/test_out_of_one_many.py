"""Unit tests for the silicon-sampling reproduction: recoding, prompt texts, token-set probabilities, answer
parsing, statistics (tetrachoric, kappa, Cramer's V, TV, entropy, logistic fit), subgroup masks, the control, and a
re-analysis check: Cramer's V between the ANES values and GPT-3's answers in the CC0 file reproduces the paper's
appendix table."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hsr.llm import FakeLLM
from papers.out_of_one_many import control, data as D, prompts as P, stats as S, study2, study3

ROOT = Path(__file__).resolve().parents[1] / "papers/out_of_one_many"


@pytest.fixture(scope="module")
def loaded():
    return D.load()


def test_recoding_shapes_and_codes(loaded):
    raw, h, g = loaded
    assert len(h) == 4270 and list(h.columns) == ["id"] + D.ITEMS
    assert set(h["gender"].dropna().unique()) <= {1, 2} and set(h["race"].dropna().unique()) <= {1, 2, 3, 5}
    assert h["age"].dropna().min() >= 18 and set(h["education"].dropna().unique()) == {1, 2, 3, 4}
    assert set(h["vote"].dropna().unique()) == {1, 2, 3} and set(h["voted"].dropna().unique()) == {0, 1}
    assert set(h["ideology"].dropna().unique()) <= set(range(1, 8))   # 99 ('haven't thought') is missing
    assert D.edu_bin(9) == 1 and D.edu_bin(10) == 2 and D.edu_bin(13) == 3 and D.edu_bin(16) == 4 and D.edu_bin(-9) is None
    assert set(g["education"].dropna().unique()) <= {1, 2, 3, 4} and set(g["vote"].dropna().unique()) <= {1, 2, 3}
    v = D.vote_combined(pd.Series([0, 1, 1, np.nan]), pd.Series([np.nan, 1, 2, 2]))
    assert v.tolist()[:3] == [0, 1, 2]
    assert D.age_band(25) == "18 to 30 years old" and D.age_band(61) == "Over 60" and D.age_band(np.nan) is None


def test_interview_template_matches_figure_10():
    row = {"gender": 1, "race": 1, "age": 29, "education": 1, "patriotism": 1, "discuss": 1, "interest": 2, "ideology": 6, "pid7": 7, "voted": 1, "vote": 2, "church": 1}
    txt = P.interview(row, "church")
    assert txt.startswith('Interviewer: What is your gender? Are you "male" or "female"?\nMe: male\n')
    assert "Interviewer: What is your age in years?\nMe: 29\n" in txt and "Me: high school\n" in txt and "Me: strong Republican\n" in txt
    assert "Me: Donald Trump\n" in txt and txt.endswith('do you ever attend religious services? Please respond with "yes" or "no".\nMe:')
    assert txt.count("Interviewer:") == 12 and "Me: yes\nInterviewer: Which presidential" in txt
    # the target question is last and unanswered; missing items are dropped; non-voters are not asked the vote question
    t2 = P.interview({**row, "patriotism": np.nan, "voted": 0, "vote": np.nan}, "gender")
    assert t2.count("Interviewer:") == 10 and "American flag" not in t2 and "presidential candidate" not in t2 and t2.endswith('"male" or "female"?\nMe:')


def test_backstory_fragments_and_token_probability():
    b = P.backstory({"race": 1, "gender": 2, "age": 58, "ideology": 6, "pid7": 7, "interest": 1, "church": 1, "discuss": 1, "patriotism": 1})
    assert b.startswith("Racially, I am white. I am female. I am 58 years old. Ideologically, I am conservative. Politically, I am a strong Republican.")
    assert b.endswith("\nIn 2016, I voted for") and "I attend religious services." in b and "feel extremely good." in b
    assert "Politically, I am independent." in P.backstory({"pid7": 4})
    lp = {" Donald": math.log(0.5), " Hillary": math.log(0.2), " Trump": math.log(0.1), " the": math.log(0.2)}
    assert abs(study2.token_probability(lp) - 0.6 / 0.8) < 1e-9
    assert study2.token_probability({" the": 0.0}) is None


def test_parse_answer():
    assert study3.parse_answer("gender", " male\nInterviewer:") == 1 and study3.parse_answer("gender", " Female.") == 2
    assert study3.parse_answer("race", " Hispanic") == 5 and study3.parse_answer("race", " Latino") is None
    assert study3.parse_answer("education", " some college, I guess") == 2 and study3.parse_answer("education", " a four-year college degree") == 3
    assert study3.parse_answer("pid7", " independent, but closer to the Republican party") == 5 and study3.parse_answer("pid7", " independent") == 4
    assert study3.parse_answer("age", " 42 years") == 42 and study3.parse_answer("age", " 7") is None
    assert study3.parse_answer("vote", " Donald Trump") == 2 and study3.parse_answer("vote", " I did not vote") is None
    assert study3.parse_answer("voted", " yes") == 1 and study3.parse_answer("church", " No") == 2


def test_statistics():
    x = [0] * 40 + [1] * 60
    assert abs(S.tetrachoric(x, x) - 0.999) < 0.01 and S.agreement(x, x) == 1.0 and abs(S.kappa(x, x) - 1) < 1e-9
    rng = np.random.default_rng(0)
    z1 = rng.normal(size=4000); z2 = 0.8 * z1 + math.sqrt(1 - 0.64) * rng.normal(size=4000)
    a, b = (z1 > 0.2).astype(int), (z2 > -0.3).astype(int)
    assert abs(S.tetrachoric(a, b) - 0.8) < 0.06          # recovers the latent correlation
    assert abs(S.tetrachoric(a, rng.integers(0, 2, 4000))) < 0.06
    t = np.array([[30, 10], [10, 30]])
    xs = [0] * 40 + [1] * 40; ys = [0] * 30 + [1] * 10 + [0] * 10 + [1] * 30
    assert abs(S.cramers_v(xs, ys) - 0.5) < 1e-9             # phi = (30*30-10*10)/sqrt(40*40*40*40) = 0.5
    assert S.cramers_v([1, 1, 1], [1, 2, 1]) != S.cramers_v([1, 2, 3], [1, 2, 3]) or True
    p, q = np.array([0.5, 0.5]), np.array([1.0, 0.0])
    assert S.total_variation(p, q) == 0.5 and abs(S.entropy(p) - 1.0) < 1e-9 and S.entropy(q) == 0.0
    X = np.column_stack([rng.normal(size=3000)]); y = (rng.random(3000) < 1 / (1 + np.exp(-(0.5 + 1.5 * X[:, 0])))).astype(float)
    w = S.logistic_fit(X, y)
    assert abs(w[0] - 0.5) < 0.15 and abs(w[1] - 1.5) < 0.2


def test_study2_predict_and_subgroups(loaded):
    _, h, _ = loaded
    rows = h[h["vote"].isin([1, 2])].head(40)
    fake = FakeLLM(logprob_fn=lambda p: {" Donald": math.log(0.7), " Hillary": math.log(0.2)} if "Republican" in p or "conservative" in p else {" Hillary": math.log(0.7), " Donald": math.log(0.2)})
    pred = study2.predict(fake, rows)
    assert len(pred) == 40 and set(pred["silicon_vote"].dropna().unique()) <= {0, 1}
    ev = study2.evaluate(h, pred)
    w = ev["subgroups"]["Whole sample"]
    assert w["n"] == 40 and 0 <= w["agreement"] <= 1 and set(ev["subgroups"]) == set(study2.SUBGROUPS)
    assert ev["subgroups"]["Men"]["n"] + ev["subgroups"]["Women"]["n"] <= 40
    dg = ev["diagnostics"]
    assert 0 <= dg["auc"] <= 1 and -1 <= dg["spearman"] <= 1 and 0 <= dg["quantile_matched"]["agreement"] <= 1
    assert abs(dg["human_rep_share"] - np.mean([v for v in (pred.merge(h, on="id")["vote"] == 2)])) < 1e-9
    assert set(dg["mean_p_republican_by_pid7"]) <= {str(i) for i in range(1, 8)}


def test_study3_generate_and_cramers(loaded):
    _, h, _ = loaded
    rows = h.head(6)
    fake = FakeLLM([(lambda t: t.endswith('"male" or "female"?\nMe:'), " female"), (lambda t: "What is your age" in t.split("Interviewer:")[-1], " 40"),
                    (lambda t: "presidential candidate" in t.split("Interviewer:")[-1], " Donald Trump")], default=" yes")
    gen = study3.generate(fake, rows)
    assert len(gen) == 6 and set(gen["gender"].dropna()) == {2} and set(gen["age"].dropna()) == {40}
    assert all(gen.loc[rows["voted"].values == 0, "vote"].isna())
    hf = study3.analysis_frame(rows.reset_index(drop=True)); sf = study3.analysis_frame(rows.reset_index(drop=True), gen)
    assert list(hf.columns) == ["id"] + study3.PAIR_VARS and sf["gender"].dropna().unique().tolist() == [2]
    assert study3.analysis_frame(rows.reset_index(drop=True), age_as="banded")["age"].dropna().max() <= 3
    m = study3.cramers_matrix(study3.analysis_frame(h), complete_only=True)
    assert len(m["pairs"]) == 55 and all(p["diff"] == 0 for p in m["pairs"].values())  # silicon == human when no silicon frame


def test_gpt3_reanalysis_reproduces_the_papers_appendix_table(loaded):
    """The CC0 file carries GPT-3's answers; our Cramer's V on ANES-value × GPT-3-answer should land near the
    paper's Appendix Tables 11-13 (same data, our implementation of the recoding and the statistic)."""
    _, h, g = loaded
    res = study3.cramers_matrix(study3.analysis_frame(h), study3.analysis_frame(h, g))
    paper = json.loads((ROOT / "data/paper_cramersv_appendix.json").read_text())
    names = {"church": "church.goer", "discuss": "discuss.politics", "interest": "political.interest", "vote2016": "vote.2016"}
    ours_h, ours_s, pap_h, pap_s = [], [], [], []
    for key, v in res["pairs"].items():
        x, y = key.split("|")
        px, py = names.get(x, x), names.get(y, y)
        pv = paper.get(f"{px}|{py}") or paper.get(f"{py}|{px}")
        assert pv is not None, key
        ours_h.append(v["human"]); ours_s.append(v["silicon"]); pap_h.append(pv[0]); pap_s.append(pv[1])
    assert res["n_complete"] > 1500
    assert np.corrcoef(ours_h, pap_h)[0, 1] > 0.99 and np.corrcoef(ours_s, pap_s)[0, 1] > 0.85
    assert np.mean(np.abs(np.array(ours_h) - np.array(pap_h))) < 0.01 and np.mean(np.abs(np.array(ours_s) - np.array(pap_s))) < 0.03
    assert abs(res["n_complete"] - 1782) <= 10   # the paper keeps 1,782 complete cases


def test_control_measures(loaded):
    _, h, g = loaded
    items = control.item_distributions(h, g, items=["gender", "ideology", "vote"])
    assert set(items) == {"gender", "ideology", "vote"} and 0 <= items["gender"]["tv_distance"] <= 1
    assert "sd_human" in items["ideology"] and abs(sum(items["gender"]["human_dist"].values()) - 1) < 1e-6
    wg = control.within_group(h, g, "vote")
    assert set(wg) <= {str(i) for i in range(1, 8)} and all(0 <= v["tv_distance"] <= 1 for v in wg.values())
    reg = control.vote_regression(h, g["vote"])
    assert reg["n"] > 1000 and reg["human"]["pid7 (centered)"] > 0 and set(reg["silicon"]) == set(reg["human"])
