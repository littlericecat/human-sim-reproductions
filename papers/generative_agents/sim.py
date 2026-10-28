"""The simulation loop (paper §4.3.1 action loop + §5 sandbox): each game-time step every agent follows its plan,
perceives who is at its place, decides whether to react, and converses; reflections fire on the importance
threshold; at the start of each day agents plan the day. Dialogue is stored in both agents' memory streams, which
is the only channel through which information (the party, the candidacy) can spread."""

from __future__ import annotations

import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from hsr.llm import LLM
from papers.generative_agents.agent import Agent, Persona
from papers.generative_agents.world import Clock, World

DEFAULT_START = dt.datetime(2023, 2, 13, 7, 0)
DAY_START, DAY_END = dt.time(7, 0), dt.time(23, 0)
MAX_TURNS = 4            # utterances per conversation (2 each)
MAX_DIALOGUES_PER_STEP = 2

DEFAULT_AGENTS = ["Isabella Rodriguez", "Maria Lopez", "Klaus Mueller", "Sam Moore", "Jennifer Moore", "Tom Moreno",
                  "Jane Moreno", "John Lin", "Eddy Lin", "Mei Lin", "Latoya Williams", "Rajiv Patel", "Wolfgang Schulz",
                  "Ayesha Khan", "Abigail Chen", "Hailey Johnson", "Francisco Lopez", "Carmen Ortiz", "Tamara Taylor",
                  "Yuriko Yamamoto", "Adam Smith", "Arthur Burton", "Ryan Park", "Giorgio Rossi", "Carlos Gomez"]


def load_personas(names: list[str], data_dir: Optional[Path] = None) -> list[Persona]:
    d = (data_dir or Path(__file__).parent / "data") / "personas"
    return [Persona.from_file(d / f"{n}.json") for n in names]


class Simulation:
    def __init__(self, personas: list[Persona], llm: LLM, embed: Callable[[str], np.ndarray], start: dt.datetime = DEFAULT_START,
                 step_minutes: int = 30, seed: int = 0, world: Optional[World] = None, log: Optional[Callable[[str], None]] = None,
                 decompose: bool = False) -> None:
        self.decompose = decompose
        names = [p.name for p in personas]
        self.agents = [Agent(p, llm, embed, known_names=names) for p in personas]
        self.by_name = {a.name: a for a in self.agents}
        self.world = world or World()
        self.clock = Clock(start, step_minutes)
        self.rng = random.Random(seed)
        self.llm = llm
        self.events: list[dict[str, Any]] = []
        self.dialogues: list[dict[str, Any]] = []
        self.party_attendance: dict[str, list[str]] = {}
        self.log = log or (lambda s: None)
        self._planned_day: dict[str, dt.date] = {}
        for a in self.agents:
            a.bootstrap(start)
            self.world.move(a.name, a.home, "at home")

    # ------------------------------------------------------------------ one step

    def step(self) -> None:
        now, now_text = self.clock.now, self.clock.text()
        if not (DAY_START <= now.time() <= DAY_END):
            self.clock.tick()
            return
        for a in self.agents:
            if self._planned_day.get(a.name) != now.date():
                a.plan_day(now.date(), DAY_START, DAY_END, decompose=self.decompose)
                self._planned_day[a.name] = now.date()
                self.events.append({"t": now.isoformat(), "agent": a.name, "event": "plan", "broad": a.broad_plan,
                                    "hourly": [(e.start.strftime("%H:%M"), e.activity, e.place) for e in a.schedule]})
        order = list(self.agents)
        self.rng.shuffle(order)
        for a in order:  # follow the plan
            e = a.current_entry(now)
            if e is not None:
                self.world.move(a.name, e.place, e.activity)
                a.observe(f"{a.name} is {e.activity} at {e.place}", now)
        talked: set[str] = set()
        dialogues = 0
        for a in order:  # perceive and react
            place = self.world.location[a.name]
            for other in self.world.agents_at(place, exclude=a.name):
                obs = f"{other} is {self.world.action.get(other, 'here')} at {place}"
                a.observe(obs, now)
                if a.name in talked or other in talked or dialogues >= MAX_DIALOGUES_PER_STEP:
                    continue
                yes, reaction = a.react(obs, other, now, now_text, self.world.action.get(a.name, ""))
                if yes:
                    self.events.append({"t": now.isoformat(), "agent": a.name, "event": "react", "to": other, "reaction": reaction})
                    if any(w in reaction.lower() for w in ("talk", "chat", "ask", "tell", "invite", "say", "greet", "discuss", "convers")) or reaction == "":
                        self.converse(a, self.by_name[other], place, now, now_text)
                        talked.update({a.name, other})
                        dialogues += 1
        for a in order:
            made = a.maybe_reflect(now)
            if made:
                self.events.append({"t": now.isoformat(), "agent": a.name, "event": "reflect", "insights": [m.text for m in made]})
        key = now.isoformat()
        if now.time() == dt.time(17, 0):
            self.party_attendance[key] = self.world.agents_at("Hobbs Cafe: cafe")
        self.clock.tick()

    def converse(self, a: Agent, b: Agent, place: str, now: dt.datetime, now_text: str) -> list[tuple[str, str]]:
        history: list[tuple[str, str]] = []
        speaker, listener = a, b
        for _ in range(MAX_TURNS):
            line = speaker.say(listener.name, history, now_text, now, place)
            if not line or line.upper().startswith("[END]"):
                break
            if any(n == speaker.name and u == line.replace("[END]", "").strip() for n, u in history):
                break  # the small model loops; a verbatim repeat ends the conversation
            history.append((speaker.name, line.replace("[END]", "").strip()))
            speaker, listener = listener, speaker
        if history:
            transcript = "\n".join(f"{n}: {u}" for n, u in history)
            for x, y in ((a, b), (b, a)):
                x.observe(f"{x.name} is conversing with {y.name} at {place}", now)
                x.memory.add("dialogue", f"Conversation between {a.name} and {b.name}:\n{transcript}", now, x.importance(transcript[:300]), with_=y.name)
                x.stats["dialogues"] += 1
            self.dialogues.append({"t": now.isoformat(), "place": place, "a": a.name, "b": b.name, "turns": history})
            self.log(f"[{now_text}] {a.name} <-> {b.name} at {place}: {len(history)} turns")
        return history

    def run(self, days: int = 2, on_step: Optional[Callable[["Simulation"], None]] = None) -> None:
        end = self.clock.now + dt.timedelta(days=days)
        while self.clock.now < end:
            self.step()
            if on_step:
                on_step(self)

    # ------------------------------------------------------------------ persistence

    def dump(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in self.events) + "\n", encoding="utf-8")
        (out_dir / "dialogues.json").write_text(json.dumps(self.dialogues, indent=1, ensure_ascii=False), encoding="utf-8")
        (out_dir / "agents.json").write_text(json.dumps([a.snapshot() for a in self.agents], ensure_ascii=False), encoding="utf-8")
