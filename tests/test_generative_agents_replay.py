"""Integration: the committed results of the real-model run are reproduced exactly by replaying its cassette
(no model needed), and the measurements in REPORT.md are the ones in results.json."""

import json
import re
from pathlib import Path

import pytest

from hsr.embed import Embedder
from hsr.llm import Cassette, ReplayLLM
from papers.generative_agents import measure
from papers.generative_agents.sim import DEFAULT_AGENTS, Simulation, load_personas

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "papers/generative_agents/results/ga-10x2"
CASSETTE = ROOT / "cassettes/ga-10x2.jsonl"
REPORT = ROOT / "papers/generative_agents/REPORT.md"


def _results():
    return json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))


def test_results_and_report_exist_and_agree():
    r = _results()
    cfg = r["config"]
    assert cfg["n"] == 10 and cfg["days"] == 2 and cfg["step_minutes"] == 60 and cfg["agents"] == DEFAULT_AGENTS[:10]
    report = REPORT.read_text(encoding="utf-8")
    d = r["diffusion"]
    for needle in (f"{d['party']['verified']} of {d['party']['n']}", f"{d['mayor']['verified']} of {d['mayor']['n']}",
                   f"{r['network']['start_density']:.3f}", f"{r['network']['end_density']:.3f}", str(r["party"]["n_invited"]), str(r["party"]["n_attended"])):
        assert needle in report, needle
    assert d["party"]["hallucinated"] + d["mayor"]["hallucinated"] + r["network"]["hallucinated"] >= 0
    assert "1.5b" in cfg["model"].lower() and r["llm_calls"] > 0


@pytest.mark.skipif(not Embedder().model_path or not Path(Embedder().model_path).exists(), reason="embedding model file not present")
def test_replaying_the_cassette_reproduces_the_committed_measurements():
    r = _results()
    cas = Cassette(CASSETTE, mode="replay")
    llm = ReplayLLM(cas, model_tag=r["config"]["model"])
    emb = Embedder()
    sim = Simulation(load_personas(r["config"]["agents"]), llm, emb.embed, step_minutes=r["config"]["step_minutes"], seed=r["config"]["seed"])
    g0 = measure.relationship_graph(sim.agents, sim.clock.now, sim.clock.text(), use_llm=True)
    sim.run(days=r["config"]["days"])
    now, text = sim.clock.now, sim.clock.text()
    d = measure.diffusion(sim.agents, now, text, {"party": "Isabella Rodriguez", "mayor": "Sam Moore"})
    g1 = measure.relationship_graph(sim.agents, now, text, use_llm=True)
    assert d["party"]["verified"] == r["diffusion"]["party"]["verified"] and d["mayor"]["verified"] == r["diffusion"]["mayor"]["verified"]
    assert abs(g0["density"] - r["network"]["start_density"]) < 1e-9 and abs(g1["density"] - r["network"]["end_density"]) < 1e-9
    assert len(sim.dialogues) == r["dialogues"] and measure.invited(sim.agents) == r["party"]["invited"]
    assert llm.calls == r["llm_calls"]
