"""One generative agent: persona, memory stream, importance scoring, reflection, planning, reacting, dialogue
turns and interview answers (paper §4). All language-model calls go through `self.llm` (an hsr.llm.LLM)."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from hsr.llm import LLM
from papers.generative_agents import prompts as P
from papers.generative_agents.memory import MemoryNode, MemoryStream
from papers.generative_agents.world import HOME_KEYWORDS, PUBLIC_PLACES, home_of, place_for_activity

WORK_KEYWORDS = ("work", "open ", "opens", "serve", "customer", "shift", "study", "studying", "class", "lecture", "research", "paper",
                 "library", "teach", "office", "counter", "store", "shop", "pub", "bar", "cafe", "project", "paint", "compose", "write")


def infer_workplace(persona: "Persona") -> Optional[str]:
    """The public place named in the persona text (e.g. 'cafe owner of Hobbs Cafe' -> 'Hobbs Cafe: cafe')."""
    text = " ".join((persona.learned, persona.currently, persona.daily_plan_req, persona.lifestyle)).lower()
    sectors = {place.split(":")[0].lower(): place for place in PUBLIC_PLACES}
    aliases = {"the willow market and pharmacy": "the willows market and pharmacy", "the willow market": "the willows market and pharmacy"}
    role = re.compile(r"(owner|keeper|student|professor|teacher|works|working|work|employee|bartender|barista|cashier|clerk|pharmacist|manager|"
                      r"librarian|staff)\b[^.]{0,40}?\b(at|of|in|for)\s+(the )?([a-z' ]+?)(?=[.,;]|\s+(who|and|where|that|with)\b|$)")
    candidates = []
    for m in role.finditer(text):
        name = aliases.get(m.group(4).strip(), m.group(4).strip())
        for sector in sectors:
            if sector in name or name in sector:
                candidates.append(sector)
    for sector in sectors:  # fall back to the first mention anywhere in the persona text
        if sector in text or any(a in text for a, s2 in aliases.items() if s2 == sector):
            candidates.append(sector)
    for sector in candidates:
        if sector == "oak hill college":
            return "Oak Hill College: library" if "research" in text or "student" in text else "Oak Hill College: classroom"
        return sectors[sector]
    if "artist" in text or "painter" in text or "animator" in text:
        return "artist's co-living space: common room"
    return None

REFLECTION_THRESHOLD = 150  # §4.2: sum of importance of the latest events
REACT_COOLDOWN_MIN = 120


@dataclasses.dataclass
class Persona:
    name: str
    first_name: str
    age: str
    innate: str
    learned: str
    currently: str
    lifestyle: str
    living_area: str
    daily_plan_req: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> "Persona":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(name=d["name"], first_name=d.get("first_name") or d["name"].split()[0], age=str(d.get("age", "")),
                   innate=d.get("innate", ""), learned=d.get("learned", ""), currently=d.get("currently", ""),
                   lifestyle=d.get("lifestyle", ""), living_area=d.get("living_area", ""), daily_plan_req=d.get("daily_plan_req", "") or "")

    @property
    def home(self) -> str:
        return home_of(self.living_area)

    def summary(self) -> str:
        return P.SUMMARY_DESCRIPTION.format(name=self.name, age=self.age, innate=self.innate, learned=self.learned,
                                            currently=self.currently, lifestyle=self.lifestyle)

    def seed_statements(self) -> list[str]:
        out = []
        for field in (self.learned, self.currently, self.lifestyle, self.daily_plan_req):
            for s in re.split(r"(?<=[.!?])\s+", field.strip()):
                if s.strip():
                    out.append(s.strip())
        return out


@dataclasses.dataclass
class PlanEntry:
    start: dt.datetime
    activity: str
    place: str


def parse_rating(text: str) -> int:
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else 3


def parse_numbered(text: str, max_items: int = 8) -> list[str]:
    items = re.findall(r"(?:^|\n)\s*(?:\d+[).:]|[-*•])\s*(.+?)(?=\n\s*(?:\d+[).:]|[-*•])|\Z)", text, re.S)
    items = [re.sub(r"^\s*\d+[).:]\s*", "", re.sub(r"\s+", " ", i)).strip().rstrip(".") for i in items]
    return [i for i in items if i][:max_items]


def parse_time(text: str) -> Optional[tuple[int, int]]:
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b", text, re.I)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower().replace(".", "")
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
        return (h % 24, mi)
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def parse_hourly(text: str) -> list[tuple[tuple[int, int], str]]:
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*(?:\d+[).]\s*)?(\d{1,2}:\d{2})\s*(?:am|pm)?\s*[-–:]\s*(.+)", line, re.I)
        if m:
            t = parse_time(line)
            if t:
                out.append((t, m.group(2).strip().rstrip(".")))
    return out


def parse_insights(text: str) -> list[tuple[str, list[int]]]:
    out = []
    for item in parse_numbered(text, 5):
        m = re.search(r"\(because of ([\d,\s]+)\)", item, re.I)
        ids = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
        clean = re.sub(r"\s*\(because of [\d,\s]+\)\s*", "", item, flags=re.I).strip()
        if clean:
            out.append((clean, ids))
    return out


CONVERSE_WORDS = ("talk", "chat", "ask", "tell", "invite", "say", "greet", "discuss", "convers", "speak", "mention", "remind", "offer", "welcome",
                  "introduce", "approach", "join", "share")
NEGATION_HEADS = ("no", "not ", "there is no need", "it is not necessary", "it's not necessary", "doesn't need", "does not need", "should not", "shouldn't")


def wants_to_converse(reaction: str) -> bool:
    """Read the free-text reaction: a conversation is intended when it names a speech act and does not open with a refusal."""
    r = reaction.strip().lower()
    if not r:
        return False
    head = r[:40]
    if any(head.startswith(n) for n in NEGATION_HEADS):
        return False
    return any(w in r for w in CONVERSE_WORDS)


class Agent:
    creative_temperature = 0.7  # plans, reactions, dialogue; scoring/parsing prompts use 0.0

    def __init__(self, persona: Persona, llm: LLM, embed: Callable[[str], np.ndarray], known_names: Optional[list[str]] = None) -> None:
        self.persona, self.llm = persona, llm
        self.memory = MemoryStream(embed)
        self.name = persona.name
        self.home = persona.home
        self.schedule: list[PlanEntry] = []
        self.broad_plan: list[str] = []
        self.yesterday: list[str] = []
        self.last_reacted: dict[str, dt.datetime] = {}
        self.known: set[str] = set()  # seeded relationships: other agents named in the persona text
        self.reflections = 0
        self.stats = {"observations": 0, "reflections": 0, "dialogues": 0, "reactions": 0}
        self._importance_cache: dict[str, int] = {}
        self._place_cache: dict[str, str] = {}
        self.workplace = infer_workplace(persona)
        if known_names:
            text = " ".join((persona.learned, persona.currently, persona.daily_plan_req))
            for other in known_names:
                if other != self.name and (other in text or other.split()[0] in text.split()):
                    self.known.add(other)

    # ---------------------------------------------------------------- memory

    def importance(self, text: str) -> int:
        if text not in self._importance_cache:
            out = self.llm.complete(P.IMPORTANCE.format(memory=text), max_tokens=4, temperature=0.0)
            self._importance_cache[text] = parse_rating(out)
        return self._importance_cache[text]

    def observe(self, text: str, now: dt.datetime, kind: str = "observation", **meta: Any) -> Optional[MemoryNode]:
        recent = [m for m in self.memory.nodes[-20:] if m.text == text and (now - m.created) <= dt.timedelta(minutes=60)]
        if recent:
            return None
        self.stats["observations"] += 1
        return self.memory.add(kind, text, now, self.importance(text), **meta)

    def bootstrap(self, now: dt.datetime) -> None:
        for s in self.persona.seed_statements():
            self.memory.add("observation", s, now, self.importance(s), seed=True)

    def retrieve_text(self, query: str, now: dt.datetime, k: int = 8) -> str:
        return "\n".join(f"- {m.text}" for m in self.memory.retrieve(query, now, k=k))

    # ---------------------------------------------------------------- reflection (§4.2)

    def maybe_reflect(self, now: dt.datetime, force: bool = False) -> list[MemoryNode]:
        if not force and self.memory.importance_since_reflection < REFLECTION_THRESHOLD:
            return []
        recent = self.memory.recent(100)
        statements = "\n".join(f"{i + 1}. {m.text}" for i, m in enumerate(recent))
        q_text = self.llm.complete(P.REFLECTION_QUESTIONS.format(statements=statements), max_tokens=120, temperature=0.0)
        questions = parse_numbered(q_text, 3) or [f"What is {self.name} focused on these days?"]
        made: list[MemoryNode] = []
        for q in questions:
            evidence = self.memory.retrieve(q, now, k=15)
            numbered = "\n".join(f"{i + 1}. {m.text}" for i, m in enumerate(evidence))
            ins = self.llm.complete(P.REFLECTION_INSIGHTS.format(name=self.name, statements=numbered), max_tokens=200, temperature=0.0)
            for insight, cites in parse_insights(ins):
                pointers = [evidence[c - 1].id for c in cites if 1 <= c <= len(evidence)]
                made.append(self.memory.add("reflection", insight, now, self.importance(insight), pointers=pointers, question=q))
        self.memory.importance_since_reflection = 0
        self.stats["reflections"] += 1
        return made

    # ---------------------------------------------------------------- planning (§4.3)

    def plan_day(self, day: dt.date, day_start: dt.time, day_end: dt.time, decompose: bool = False) -> list[PlanEntry]:
        """Top-down daily plan (§4.3). `decompose=True` adds the hourly refinement inside each broad item's window;
        it is off by default because the 1.5B local model's decompositions were incoherent (see REPORT.md)."""
        p = self.persona
        yesterday = ("On " + (day - dt.timedelta(days=1)).strftime("%A %B %d") + ", " + p.first_name + " " +
                     "; ".join(self.yesterday) + ".\n") if self.yesterday else ""
        today = day.strftime("%A %B %d")
        broad_txt = "1)" + self.llm.complete(P.DAILY_PLAN.format(name=p.name, age=p.age, innate=p.innate, learned=p.learned, currently=p.currently,
                                                                  lifestyle=p.lifestyle, yesterday=yesterday, today=today, first=p.first_name),
                                             max_tokens=220, temperature=self.creative_temperature)
        self.broad_plan = parse_numbered(broad_txt) or [f"go about {p.first_name}'s usual day"]
        start, end = day_start.strftime("%H:%M"), day_end.strftime("%H:%M")
        hourly: list[tuple[tuple[int, int], str]] = []
        if decompose:
            hourly_txt = self.llm.complete(P.HOURLY_PLAN.format(summary=p.summary(), today=today, first=p.first_name,
                                                                broad="\n".join(f"{i + 1}) {b}" for i, b in enumerate(self.broad_plan)), start=start, end=end),
                                           max_tokens=360, temperature=0.0)
            hourly = parse_hourly(hourly_txt)
        anchors = self.anchor_times(self.broad_plan, day_start)
        items: list[tuple[tuple[int, int], str]] = []
        for i, ((h, mi), b) in enumerate(anchors):  # the broad item anchors its window; hourly lines inside the window refine it
            nxt = anchors[i + 1][0] if i + 1 < len(anchors) else (day_end.hour, day_end.minute)
            inside = [(t, act) for t, act in hourly if (h, mi) <= t < nxt]
            items.append(((h, mi), b))
            items += [(t, act) for t, act in inside if t != (h, mi)]
        entries = []
        for (h, mi), activity in sorted(items, key=lambda x: x[0]):
            entries.append(PlanEntry(dt.datetime.combine(day, dt.time(h, mi)), activity, self.choose_place(activity)))
        self.schedule = [e for e in entries if day_start <= e.start.time() <= day_end] or entries
        self.memory.add("plan", f"{p.first_name}'s plan for {today}: " + "; ".join(self.broad_plan), dt.datetime.combine(day, day_start), 5)
        self.yesterday = list(self.broad_plan)
        return self.schedule

    @staticmethod
    def anchor_times(broad: list[str], day_start: dt.time) -> list[tuple[tuple[int, int], str]]:
        """Each broad item with its stated time; items without a time are spread evenly up to the next stated one."""
        times: list[Optional[tuple[int, int]]] = [parse_time(b) for b in broad]
        if not times or times[0] is None:
            if times:
                times[0] = (day_start.hour, day_start.minute)
        for i, t in enumerate(times):
            if t is None:
                prev = times[i - 1]
                times[i] = ((prev[0] + 2) % 24, prev[1])
        out, last = [], (-1, -1)
        for t, b in zip(times, broad):  # keep the sequence monotone
            if t <= last:
                t = ((last[0] + 1) % 24, last[1])
            out.append((t, b))
            last = t
        return out

    def choose_place(self, activity: str) -> str:
        """Home for routine activities, the persona's workplace for work-like ones, otherwise the model picks from
        the list (accepted only when it names a listed place exactly), else a keyword fallback, else home."""
        if activity not in self._place_cache:
            a = activity.lower()
            named = next((pl for pl in PUBLIC_PLACES if pl.split(":")[0].lower() in a), None)
            if named:  # an explicitly named place wins ("go to Hobbs Cafe")
                place = named if named.split(":")[0] != "Oak Hill College" else (self.workplace if (self.workplace or "").startswith("Oak Hill") else named)
            elif any(k in a for k in HOME_KEYWORDS):
                place = self.home
            elif self.workplace and any(k in a for k in WORK_KEYWORDS) and not any(k in a for k in ("park", "walk", "grocer", "market", "pharmacy")):
                place = self.workplace
            else:
                places = PUBLIC_PLACES + [self.home]
                out = self.llm.complete(P.CHOOSE_LOCATION.format(name=self.name, activity=activity, home=self.home, places="; ".join(places)),
                                        max_tokens=24, temperature=0.0).strip().splitlines()[0].strip().strip(".\"'") if True else ""
                match = next((pl for pl in places if pl.lower() == out.lower()), None)
                place = match or place_for_activity(activity, self.workplace or self.home)
            self._place_cache[activity] = place
        return self._place_cache[activity]

    def current_entry(self, now: dt.datetime) -> Optional[PlanEntry]:
        current = None
        for e in self.schedule:
            if e.start <= now:
                current = e
        return current or (self.schedule[0] if self.schedule else None)

    def replan_from(self, now: dt.datetime, activity: str, place: str, minutes: int = 60) -> None:
        """§4.3.1: regenerate the plan from the moment of a reaction — here, insert the reaction and resume."""
        self.schedule = [e for e in self.schedule if e.start <= now or e.start >= now + dt.timedelta(minutes=minutes)]
        self.schedule.append(PlanEntry(now, activity, place))
        self.schedule.sort(key=lambda e: e.start)

    # ---------------------------------------------------------------- reacting (§4.3.1)

    def react(self, observation: str, other: str, now: dt.datetime, now_text: str, status: str) -> tuple[bool, str]:
        last = self.last_reacted.get(other)
        if last is not None and (now - last) < dt.timedelta(minutes=REACT_COOLDOWN_MIN):
            return False, "cooldown"
        self.last_reacted[other] = now
        mems = self.memory.retrieve(f"What is {self.name}'s relationship with {other}?", now, k=4) + self.memory.retrieve(observation, now, k=4)
        seen, lines = set(), []
        for m in mems:
            if m.id not in seen:
                seen.add(m.id)
                lines.append(f"- {m.text}")
        context = self.llm.complete(P.CONTEXT_SUMMARY.format(name=self.name, memories="\n".join(lines), other=other, observation=observation),
                                    max_tokens=80, temperature=0.0)
        out = self.llm.complete(P.REACT.format(summary=self.persona.summary(), now=now_text, name=self.name, status=status, observation=observation, context=context),
                                max_tokens=70, temperature=self.creative_temperature)
        yes = wants_to_converse(out)
        if yes:
            self.stats["reactions"] += 1
        return yes, out.strip()

    # ---------------------------------------------------------------- dialogue (§4.3.2) and interviews (§6)

    def say(self, other: str, history: list[tuple[str, str]], now_text: str, now: dt.datetime, place: str) -> str:
        last = history[-1][1] if history else f"{self.name} wants to talk to {other}"
        memories = self.retrieve_text(f"{other}. {last}", now, k=6)
        hist = "\n".join(f"{n}: {u}" for n, u in history) if history else f"{self.name} starts the conversation."
        return self.llm.complete(P.DIALOGUE_TURN.format(summary=self.persona.summary(), now=now_text, name=self.name, place=place, memories=memories,
                                                        other=other, history=hist), max_tokens=70, temperature=self.creative_temperature).strip()

    def interview(self, question: str, now: dt.datetime, now_text: str, k: int = 8, max_tokens: int = 80) -> str:
        memories = self.retrieve_text(question, now, k=k)
        return self.llm.complete(P.INTERVIEW.format(summary=self.persona.summary(), now=now_text, name=self.name, memories=memories, question=question),
                                 max_tokens=max_tokens, temperature=0.0).strip()

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "home": self.home, "known_seed": sorted(self.known), "stats": dict(self.stats),
                "schedule": [{"start": e.start.isoformat(), "activity": e.activity, "place": e.place} for e in self.schedule],
                "memory": self.memory.as_list()}
