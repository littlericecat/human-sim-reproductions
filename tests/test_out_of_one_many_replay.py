"""Integration: the committed silicon-sampling results are reproduced from the cassette without a model, and the
report quotes the numbers in results.json."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers/out_of_one_many/results/ooom"
CASSETTE = ROOT / "cassettes/ooom.jsonl"
REPORT = ROOT / "papers/out_of_one_many/REPORT.md"


def _results():
    return json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))


def test_results_and_report_agree():
    r = _results()
    report = REPORT.read_text(encoding="utf-8")
    w = r["study2"]["subgroups"]["Whole sample"]
    for needle in (f"{w['tetrachoric']:.2f}", f"{w['agreement']:.2f}", str(w["n"]), str(r["config"]["n3"]),
                   f"{r['study3_local']['mean_diff_human_minus_silicon']:.3f}", f"{r['study3_local']['correlation']:.2f}",
                   f"{r['study3_gpt3_reanalysis']['correlation']:.2f}"):
        assert needle in report, needle
    assert r["config"]["n2"] == w["n"] + r["study2"].get("n_no_candidate_token", 0) or r["config"]["n2"] >= w["n"]
    assert "1.5b" in r["config"]["model"].lower()


def test_replaying_the_cassette_reproduces_results(tmp_path):
    r = _results()
    out = tmp_path / "replay"
    proc = subprocess.run([sys.executable, "-m", "papers.out_of_one_many.run", "--n2", str(r["config"]["n2"]), "--n3", str(r["config"]["n3"]),
                           "--seed", str(r["config"]["seed"]), "--out", str(out), "--cassette", str(CASSETTE), "--replay"],
                          cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stderr[-2000:]
    rep = json.loads((out / "results.json").read_text())
    for name in ("Whole sample", "Strong partisans", "Independents", "Liberals", "Over 60"):
        a, b = rep["study2"]["subgroups"][name], r["study2"]["subgroups"][name]
        assert a["n"] == b["n"] and abs(a.get("tetrachoric", 0) - b.get("tetrachoric", 0)) < 1e-9 and abs(a.get("agreement", 0) - b.get("agreement", 0)) < 1e-9
    assert rep["study3_local"]["pairs"] == r["study3_local"]["pairs"] and rep["study3_local_parse_rate"] == r["study3_local_parse_rate"]
    assert rep["control"]["local"]["items"]["vote"] == r["control"]["local"]["items"]["vote"]
    assert rep["llm_calls"] == r["llm_calls"]
