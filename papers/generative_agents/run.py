"""Run the Smallville reproduction with a local model and write results.

  python -m papers.generative_agents.run --n 12 --days 2 --step-minutes 30 --out runs/ga-12x2 --cassette cassettes/ga-12x2.jsonl
  python -m papers.generative_agents.run ... --replay      # reproduce the numbers from the cassette, no model

Outputs: results.json (measurements of §7.1 + call counts), events.jsonl, dialogues.json, agents.json (memory streams).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from hsr.embed import Embedder
from hsr.llm import Cassette, LocalLLM, ReplayLLM
from papers.generative_agents import measure
from papers.generative_agents.sim import DEFAULT_AGENTS, Simulation, load_personas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--step-minutes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cassette", required=True)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cassette = Cassette(a.cassette, mode="replay" if a.replay else "record")
    llm = ReplayLLM(cassette, model_tag=Path(a.model or LocalLLM().model_path).name) if a.replay else LocalLLM(**({"model_path": a.model} if a.model else {}), cassette=cassette)
    emb = Embedder()
    names = DEFAULT_AGENTS[: a.n]
    t0 = time.time()
    log_f = (out / "log.txt").open("a", encoding="utf-8")

    def log(s: str) -> None:
        log_f.write(s + "\n")
        log_f.flush()

    sim = Simulation(load_personas(names), llm, emb.embed, step_minutes=a.step_minutes, seed=a.seed, log=log)
    start_text = sim.clock.text()
    graph0 = measure.relationship_graph(sim.agents, sim.clock.now, start_text, use_llm=True)
    log(f"start density {graph0['density']:.3f} ({len(graph0['edges'])} edges)")
    steps = 0

    def on_step(s: Simulation) -> None:
        nonlocal steps
        steps += 1
        if steps % 8 == 0:
            log(f"step {steps} {s.clock.text()} | llm calls {llm.calls} | dialogues {len(s.dialogues)} | {time.time() - t0:.0f}s")

    sim.run(days=a.days, on_step=on_step)
    now, now_text = sim.clock.now, sim.clock.text()
    diff = measure.diffusion(sim.agents, now, now_text, {"party": "Isabella Rodriguez", "mayor": "Sam Moore"})
    graph1 = measure.relationship_graph(sim.agents, now, now_text, use_llm=True)
    inv = measure.invited(sim.agents)
    party_key = dt.datetime(2023, 2, 14, 17, 0).isoformat()
    attended = sim.party_attendance.get(party_key, [])
    results = {
        "config": {"agents": names, "n": len(names), "days": a.days, "step_minutes": a.step_minutes, "seed": a.seed, "model": llm.model_tag,
                   "embedder": Path(emb.model_path).name, "start": sim.clock.now.isoformat()},
        "llm_calls": llm.calls, "cassette_entries": len(cassette), "elapsed_s": round(time.time() - t0, 1),
        "diffusion": diff,
        "network": {"start_density": graph0["density"], "end_density": graph1["density"], "start_edges": len(graph0["edges"]), "end_edges": len(graph1["edges"]),
                    "responses": graph0["responses"] + graph1["responses"], "hallucinated": graph0["hallucinated"] + graph1["hallucinated"],
                    "start_graph": graph0, "end_graph": graph1},
        "party": {"invited": inv, "n_invited": len(inv), "attended_17h_feb14": attended, "n_attended": len([x for x in attended if x != "Isabella Rodriguez"]),
                  "attendance_by_step": sim.party_attendance},
        "agent_stats": {ag.name: ag.stats for ag in sim.agents}, "dialogues": len(sim.dialogues),
    }
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    sim.dump(out)
    print(json.dumps({k: results[k] for k in ("llm_calls", "elapsed_s", "dialogues")}, indent=1))
    print("party:", diff["party"]["verified"], "/", diff["party"]["n"], "| mayor:", diff["mayor"]["verified"], "/", diff["mayor"]["n"],
          "| density", round(graph0["density"], 3), "->", round(graph1["density"], 3), "| invited", len(inv), "attended", results["party"]["n_attended"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
