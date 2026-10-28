"""The end-to-end measurements of paper §7.1: information diffusion (party, candidacy), relationship network
density from "Do you know of <name>?" interviews, hallucination checks against the memory stream, and party
attendance. Interview answers are labelled by rules that mirror the paper's manual yes/no coding."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

from papers.generative_agents.agent import Agent

PARTY_Q = "Did you know there is a Valentine's Day party?"
MAYOR_Q = "Do you know who is running for mayor?"
NEGATIVE = ("no", "not sure", "i'm not sure", "i am not sure", "i don't know", "i do not know", "i'm not aware", "i am not aware", "i haven't heard",
            "i have not heard", "not that i know", "i don't think", "unfortunately")


def labelled_yes(answer: str, keywords: tuple[str, ...]) -> bool:
    a = answer.strip().lower()
    if not a:
        return False
    head = a[:60]
    if any(head.startswith(n) for n in NEGATIVE):
        return False
    return head.startswith("yes") or any(k in a for k in keywords)


def knows_party(answer: str) -> bool:
    return labelled_yes(answer, ("valentine", "party at hobbs", "isabella"))


def knows_mayor(answer: str) -> bool:
    return labelled_yes(answer, ("sam moore", "sam ", "moore"))


def has_source(agent: Agent, pattern: str) -> bool:
    """The paper verified every 'yes' by locating the dialogue in the memory stream that carried the information."""
    rx = re.compile(pattern, re.I)
    return any(rx.search(m.text) for m in agent.memory.nodes if m.kind in ("dialogue", "observation", "reflection"))


def diffusion(agents: list[Agent], now: dt.datetime, now_text: str, originators: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for topic, (question, label, pattern) in {"party": (PARTY_Q, knows_party, r"valentine|party"), "mayor": (MAYOR_Q, knows_mayor, r"mayor|election|running for")}.items():
        rows = []
        for a in agents:
            ans = a.interview(question, now, now_text)
            yes = label(ans)
            verified = yes and (has_source(a, pattern) or a.name == originators.get(topic))
            rows.append({"agent": a.name, "answer": ans, "yes": yes, "verified": verified})
        n = len(agents)
        out[topic] = {"question": question, "rows": rows, "n": n, "yes": sum(r["yes"] for r in rows), "verified": sum(r["verified"] for r in rows),
                      "hallucinated": sum(r["yes"] and not r["verified"] for r in rows), "share_verified": sum(r["verified"] for r in rows) / n}
    return out


def knows_person(answer: str) -> bool:
    return labelled_yes(answer, ())


def relationship_graph(agents: list[Agent], now: dt.datetime, now_text: str, use_llm: bool = True) -> dict[str, Any]:
    """Mutual knowledge graph. An affirmative answer counts only if the name occurs in the agent's memory stream
    (the paper's hallucination check); the raw count is kept for the hallucination rate."""
    names = [a.name for a in agents]
    knows: dict[tuple[str, str], bool] = {}
    responses = hallucinated = 0
    for a in agents:
        for other in names:
            if other == a.name:
                continue
            in_memory = any(other in m.text or other.split()[0] in m.text.split() for m in a.memory.nodes) or other in a.known
            if use_llm:
                ans = a.interview(f"Do you know of {other}?", now, now_text, k=4, max_tokens=40)
                yes = knows_person(ans)
                responses += 1
                if yes and not in_memory:
                    hallucinated += 1
                knows[(a.name, other)] = yes and in_memory
            else:
                knows[(a.name, other)] = in_memory
    edges = {tuple(sorted((x, y))) for (x, y), v in knows.items() if v and knows.get((y, x))}
    n = len(names)
    density = 2 * len(edges) / (n * (n - 1)) if n > 1 else 0.0
    return {"n": n, "edges": sorted(edges), "density": density, "responses": responses, "hallucinated": hallucinated,
            "hallucination_rate": hallucinated / responses if responses else 0.0}


def invited(agents: list[Agent], host: str = "Isabella Rodriguez") -> list[str]:
    """Agents whose memory stream carries a party mention from a dialogue (the host's invitation reaching them)."""
    out = []
    for a in agents:
        if a.name == host:
            continue
        if any(m.kind == "dialogue" and re.search(r"valentine|party", m.text, re.I) for m in a.memory.nodes):
            out.append(a.name)
    return out
