"""Run the silicon-sampling reproduction with a local model.

  python -m papers.out_of_one_many.run --n2 1500 --n3 250 --seed 0 --out runs/ooom --cassette cassettes/ooom.jsonl
  python -m papers.out_of_one_many.run ... --replay          # reproduce results.json from the cassette, no model

Writes results.json with: Study 2 (token-probability vote prediction; tetrachoric / kappa / agreement by subgroup, the
paper's 2016 Table 1 alongside), Study 3 (Cramer's V pairs for human, local-model silicon and — re-analysed from
the CC0 file — GPT-3's own answers; the paper's appendix values alongside), and the Bisbee distribution control.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from hsr.llm import Cassette, LocalLLM, ReplayLLM
from papers.out_of_one_many import control, data as D, study2, study3


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return df if n >= len(df) else df.sample(n=n, random_state=seed).sort_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n2", type=int, default=1500, help="respondents for Study 2 (vote probabilities)")
    ap.add_argument("--n3", type=int, default=250, help="respondents for Study 3 (12 completions each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cassette", required=True)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cassette = Cassette(a.cassette, mode="replay" if a.replay else "record")
    llm = ReplayLLM(cassette, model_tag=Path(a.model or LocalLLM().model_path).name) if a.replay else \
        LocalLLM(**({"model_path": a.model} if a.model else {}), cassette=cassette, n_ctx=1024, logits_all=True)
    t0 = time.time()
    raw, human, gpt3 = D.load()
    log = (out / "log.txt").open("a", encoding="utf-8")

    # ---- Study 2: two-party voters with a vote and at least the partisanship/ideology fields present
    eligible = human[human["vote"].isin([1, 2])]
    s2_rows = sample(eligible, a.n2, a.seed)
    log.write(f"study 2: {len(s2_rows)} of {len(eligible)} two-party voters\n"); log.flush()
    s2_pred = study2.predict(llm, s2_rows)
    s2 = study2.evaluate(human, s2_pred)
    s2["n_prompted"] = int(len(s2_rows))
    s2["n_no_candidate_token"] = int(s2_pred["p_republican"].isna().sum())
    log.write(f"study 2 done: {llm.calls} calls, {time.time() - t0:.0f}s\n"); log.flush()
    s2_pred.to_csv(out / "study2_predictions.csv", index=False)

    # ---- Study 3: local-model silicon sample (a plain instance: logits_all is only needed for Study 2 and slows generation)
    calls_s2 = llm.calls
    if not a.replay:
        llm.close()
        llm = LocalLLM(**({"model_path": a.model} if a.model else {}), cassette=cassette, n_ctx=1024, logits_all=False)
    s3_rows = sample(human, a.n3, a.seed + 1)
    s3_gen = study3.generate(llm, s3_rows, seed=a.seed)
    s3_gen.to_csv(out / "study3_generated.csv", index=False)
    h_frame = study3.analysis_frame(s3_rows.reset_index(drop=True))
    s_frame = study3.analysis_frame(s3_rows.reset_index(drop=True), s3_gen.set_index("id").loc[s3_rows["id"]].reset_index())
    s3_local = study3.cramers_matrix(h_frame, s_frame)
    parse_rate = {item: float(s3_gen[item].notna().mean()) for item in D.ITEMS}
    total_calls = calls_s2 + (llm.calls if not a.replay else 0) if not a.replay else llm.calls
    log.write(f"study 3 done: {total_calls} calls, {time.time() - t0:.0f}s\n"); log.flush()

    # ---- Study 3 re-analysis of GPT-3's own answers (CC0 file; no model)
    gh = study3.analysis_frame(human)
    gs = study3.analysis_frame(human, gpt3)
    s3_gpt3 = study3.cramers_matrix(gh, gs)
    paper = json.loads((Path(__file__).parent / "data" / "paper_cramersv_appendix.json").read_text())
    table1 = json.loads((Path(__file__).parent / "data" / "paper_table1_2016.json").read_text())

    # ---- Bisbee control on the local-model silicon sample (and on GPT-3's answers for reference)
    s3_rows_r = s3_rows.reset_index(drop=True)
    s3_gen_r = s3_gen.set_index("id").loc[s3_rows_r["id"]].reset_index()
    ctrl = {"local": {"items": control.item_distributions(s3_rows_r, s3_gen_r), "vote_by_pid7": control.within_group(s3_rows_r, s3_gen_r, "vote"),
                      "ideology_by_pid7": control.within_group(s3_rows_r, s3_gen_r, "ideology"), "vote_regression": control.vote_regression(s3_rows_r, s3_gen_r["vote"])},
            "gpt3": {"items": control.item_distributions(human, gpt3), "vote_by_pid7": control.within_group(human, gpt3, "vote"),
                     "ideology_by_pid7": control.within_group(human, gpt3, "ideology"), "vote_regression": control.vote_regression(human, gpt3["vote"])}}

    results = {"config": {"n2": int(len(s2_rows)), "n3": int(len(s3_rows)), "seed": a.seed, "model": llm.model_tag, "rows_in_file": int(len(human))},
               "llm_calls": total_calls, "cassette_entries": len(cassette), "elapsed_s": round(time.time() - t0, 1),
               "study2": s2, "paper_table1_2016": table1["rows"],
               "study3_local": s3_local, "study3_local_parse_rate": parse_rate, "study3_gpt3_reanalysis": s3_gpt3, "paper_cramersv": paper,
               "control": ctrl}
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    w = s2["subgroups"]["Whole sample"]
    print(json.dumps({"calls": total_calls, "elapsed_s": results["elapsed_s"]}))
    print(f"study 2 whole sample: n={w['n']} tetrachoric={w.get('tetrachoric', float('nan')):.2f} agreement={w.get('agreement', float('nan')):.2f} (paper 2016: 0.92 / 0.87)")
    print(f"study 3 local: n_complete={s3_local['n_complete']} mean diff={s3_local['mean_diff_human_minus_silicon']:.3f} corr={s3_local['correlation']:.2f} | gpt3 re-analysis: n={s3_gpt3['n_complete']} mean diff={s3_gpt3['mean_diff_human_minus_silicon']:.3f} corr={s3_gpt3['correlation']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
