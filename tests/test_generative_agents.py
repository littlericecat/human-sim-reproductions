"""Unit tests for the Generative Agents reproduction: retrieval maths, parsers, reflection trigger and tree,
planning fallbacks, reaction cooldown, dialogue storage, the §7.1 measurement rules, and a scripted 3-agent
simulation with a fake model in which the party invitation spreads by dialogue only."""

import datetime as dt
import re

import numpy as np
import pytest

from hsr.embed import BagOfWordsEmbedder
from hsr.llm import FakeLLM
from papers.generative_agents import measure
from papers.generative_agents.agent import Agent, Persona, parse_hourly, parse_insights, parse_numbered, parse_rating, parse_time
from papers.generative_agents.memory import DECAY, MemoryStream, minmax
from papers.generative_agents.sim import Simulation, load_personas
from papers.generative_agents.world import World, home_of, load_places, place_for_activity

T0 = dt.datetime(2023, 2, 13, 7, 0)
EMB = BagOfWordsEmbedder().embed


def _agent(name="Isabella Rodriguez", llm=None, known=None):
    p = load_personas([name])[0]
    return Agent(p, llm or FakeLLM(default="3"), EMB, known_names=known)


# ---------------------------------------------------------------- memory & retrieval (§4.1)

def test_retrieval_scores_recency_importance_relevance():
    ms = MemoryStream(EMB)
    old = ms.add("observation", "Isabella is setting out the pastries", T0 - dt.timedelta(hours=200), 2)
    imp = ms.add("observation", "Isabella's cafe was robbed last night", T0 - dt.timedelta(hours=200), 10)
    rel = ms.add("observation", "Maria is studying for a chemistry test while drinking coffee", T0 - dt.timedelta(hours=200), 2)
    now = T0
    top = ms.retrieve("Maria is studying chemistry", now, k=1)
    assert top[0] is rel and rel.last_accessed == now            # relevance wins on a relevant query, access time updated
    scores = {m.id: s for s, m in ms.scores("something unrelated entirely", now)}
    assert scores[imp.id] > scores[old.id]                          # importance separates otherwise equal memories
    assert abs(DECAY ** 200 - 0.3670) < 1e-3                        # 0.995^hours (paper's decay factor)
    assert minmax([1, 1, 1]) == [1.0, 1.0, 1.0] and minmax([0, 5, 10]) == [0.0, 0.5, 1.0]


def test_importance_is_asked_once_per_text_and_parsed():
    llm = FakeLLM([(lambda t: "poignancy" in t, "Rating: 8")])
    a = _agent(llm=llm)
    assert a.importance("asking your crush out on a date") == 8
    assert a.importance("asking your crush out on a date") == 8 and llm.calls == 1
    assert parse_rating("7") == 7 and parse_rating("10/10") == 10 and parse_rating("no idea") == 3
    assert a.observe("x", T0) is not None and a.observe("x", T0 + dt.timedelta(minutes=10)) is None  # de-duplicated within the hour


# ---------------------------------------------------------------- parsers

def test_parsers():
    assert parse_numbered("1) wake up at 7:00 am\n2) go to the cafe at 8:00 am\n3) sleep at 10 pm.") == ["wake up at 7:00 am", "go to the cafe at 8:00 am", "sleep at 10 pm"]
    assert parse_time("wake up at 7:00 am") == (7, 0) and parse_time("sleep at 10 pm") == (22, 0) and parse_time("at 12:30 pm") == (12, 30)
    assert parse_time("12 am") == (0, 0) and parse_time("13:45 - lunch") == (13, 45) and parse_time("nothing") is None
    assert parse_hourly("07:00 - wake up\n08:00 - open the cafe\nnot a line\n17:00 - party at Hobbs Cafe") == [((7, 0), "wake up"), ((8, 0), "open the cafe"), ((17, 0), "party at Hobbs Cafe")]
    assert parse_insights("1. Klaus is dedicated to research (because of 1, 2, 8)\n2. Klaus likes Maria (because of 3)\n3. plain insight") == \
        [("Klaus is dedicated to research", [1, 2, 8]), ("Klaus likes Maria", [3]), ("plain insight", [])]


# ---------------------------------------------------------------- reflection (§4.2)

def test_reflection_fires_on_importance_threshold_and_cites_evidence():
    llm = FakeLLM([(lambda t: "salient high-level questions" in t, "1. What is Klaus passionate about?\n2. Who is Klaus close to?\n3. What does Klaus do?"),
                   (lambda t: "high-level insights" in t, "1. Klaus Mueller is dedicated to his research (because of 1, 2)\n2. Klaus likes the library (because of 3)"),
                   (lambda t: "poignancy" in t, "5")])
    a = _agent("Klaus Mueller", llm=llm)
    assert a.maybe_reflect(T0) == []                                   # below threshold
    for i in range(30):
        a.observe(f"Klaus Mueller is writing his research paper, page {i}", T0 + dt.timedelta(minutes=i))
    assert a.memory.importance_since_reflection >= 150
    made = a.maybe_reflect(T0 + dt.timedelta(hours=1))
    assert len(made) == 6 and all(m.kind == "reflection" for m in made)   # 3 questions x 2 insights
    assert made[0].pointers and all(a.memory.nodes[p].kind == "observation" for p in made[0].pointers)
    assert a.memory.importance_since_reflection == 0 and a.stats["reflections"] == 1


# ---------------------------------------------------------------- planning (§4.3)

def test_plan_day_parses_broad_and_hourly_and_places():
    llm = FakeLLM([
        (lambda t: "plan today in broad strokes" in t, " wake up and complete the morning routine at 7:00 am\n2) open Hobbs Cafe at 8:00 am\n3) decorate the cafe for the party at 2:00 pm\n4) go to bed at 10:00 pm"),
        (lambda t: "Break the plan into hourly" in t, "07:00 - wake up and morning routine\n08:00 - open Hobbs Cafe and serve customers\n14:00 - decorate the cafe for the Valentine's Day party\n22:00 - go to bed"),
        (lambda t: "Which single place" in t and "serve customers" in t, "Hobbs Cafe: cafe"),
        (lambda t: "Which single place" in t and "decorate" in t, "Hobbs Cafe: cafe"),
        (lambda t: "Which single place" in t, "somewhere unknown"),
        (lambda t: "poignancy" in t, "2")])
    a = _agent(llm=llm)
    sched = a.plan_day(T0.date(), dt.time(7, 0), dt.time(23, 0), decompose=True)
    # broad items anchor the day; hourly lines at the same times do not duplicate them
    assert [e.activity for e in sched] == ["wake up and complete the morning routine at 7:00 am", "open Hobbs Cafe at 8:00 am", "decorate the cafe for the party at 2:00 pm", "go to bed at 10:00 pm"]
    assert [e.place for e in sched] == [a.home, "Hobbs Cafe: cafe", "Hobbs Cafe: cafe", a.home]
    assert a.current_entry(T0.replace(hour=9)).activity == "open Hobbs Cafe at 8:00 am"
    assert a.current_entry(T0.replace(hour=6)).activity == "wake up and complete the morning routine at 7:00 am"
    assert a.memory.nodes[-1].kind == "plan" and a.yesterday == a.broad_plan
    b = _agent(llm=llm)
    plain = b.plan_day(T0.date(), dt.time(7, 0), dt.time(23, 0))          # default: broad anchors only, no hourly call
    assert [e.activity for e in plain] == [e.activity for e in sched]
    assert sum("Break the plan" in x for x in llm.log) == 1   # only the decompose=True plan asked for the hourly refinement
    a.replan_from(T0.replace(hour=9), "talk to Maria about the party", "Hobbs Cafe: cafe")
    assert a.current_entry(T0.replace(hour=9, minute=5)).activity == "talk to Maria about the party"


def test_plan_falls_back_when_the_model_returns_no_hourly_lines():
    llm = FakeLLM([(lambda t: "broad strokes" in t, " wake up at 8:00 am\n2) paint at the studio\n3) sleep at 11 pm"), (lambda t: "poignancy" in t, "2")], default="")
    a = _agent("Rajiv Patel", llm=llm)
    sched = a.plan_day(T0.date(), dt.time(7, 0), dt.time(23, 0))
    assert len(sched) == 3 and sched[0].start.hour == 8 and sched[-1].start.hour == 23


def test_world_places_and_fallbacks():
    places = load_places()
    assert "Hobbs Cafe" in places and "cafe" in places["Hobbs Cafe"] and len(places) >= 18
    assert home_of("the Ville:artist's co-living space:Abigail Chen's room") == "artist's co-living space: Abigail Chen's room"
    assert place_for_activity("have lunch at the cafe", "home") == "Hobbs Cafe: cafe"
    assert place_for_activity("go to bed", "home") == "home" and place_for_activity("do taxes", "home") == "home"
    w = World()
    w.move("A", "Hobbs Cafe: cafe", "x"); w.move("B", "Hobbs Cafe: cafe", "y"); w.move("C", "Johnson Park: park", "z")
    assert w.agents_at("Hobbs Cafe: cafe", exclude="A") == ["B"]


# ---------------------------------------------------------------- reacting and dialogue

def test_react_cooldown_and_parsing():
    from papers.generative_agents.agent import wants_to_converse
    llm = FakeLLM([(lambda t: "Should" in t, "Yes, Isabella could talk to Maria and invite her to the party."), (lambda t: "summarize" in t, "Maria is a friend."), (lambda t: "poignancy" in t, "3")])
    a = _agent(llm=llm)
    yes, reaction = a.react("Maria Lopez is drinking coffee at Hobbs Cafe: cafe", "Maria Lopez", T0, "t", "serving")
    assert yes and "invite" in reaction and a.stats["reactions"] == 1
    assert a.react("again", "Maria Lopez", T0 + dt.timedelta(minutes=30), "t", "serving") == (False, "cooldown")
    llm2 = FakeLLM([(lambda t: "Should" in t, "No, there is no need to react; Klaus is just having lunch."), (lambda t: "summarize" in t, "n/a"), (lambda t: "poignancy" in t, "3")])
    yes2, _ = _agent(llm=llm2).react("x", "Y", T0, "t", "s")
    assert not yes2
    assert wants_to_converse("Isabella should greet Klaus and ask how his research is going.")
    assert not wants_to_converse("There is no need to react; she continues decorating.") and not wants_to_converse("No.") and not wants_to_converse("")
    assert not wants_to_converse("She keeps working on the decorations.")


def test_measurement_labels():
    assert measure.knows_party("Yes, Isabella invited me to a Valentine's Day party at Hobbs Cafe.")
    assert measure.knows_party("Isabella Rodriguez is hosting a Valentine's Day party on the 14th.")
    assert not measure.knows_party("No, I did not know there was a Valentine's day party.")
    assert not measure.knows_party("I'm not sure if there is a party.")
    assert measure.knows_mayor("I know that Sam Moore has expressed interest in running for local mayor.")
    assert not measure.knows_mayor("I'm not sure who is running for the election.")
    assert not measure.knows_mayor("Sam Moore: Yes, I know who is running for mayor. It's Isabella Rodriguez.")   # names the wrong person
    assert not measure.knows_mayor("John Lin: Yes, I'm aware that there are several candidates running for the local mayor election.")
    assert not measure.knows_party("John Lin: Yes, I did. It's a great opportunity to celebrate love and relationships.")  # no party named
    assert measure.knows_person("Yes, I know Maria Lopez. She is a student.") and not measure.knows_person("No, I don't know who that is.")
    # answers seen in the real run: a name prefix, a refusal, a 'not aware', a generic Valentine's remark
    assert not measure.knows_party("Klaus Mueller: No, I didn't know there was a Valentine's Day party.")
    assert not measure.knows_party("Jennifer Moore: I'm sorry, but I don't recall any specific Valentine's Day party mentioned in my plans.")
    assert not measure.knows_party("Mei Lin: I'm not aware of a Valentine's Day party. I'm not particularly interested in such events.")
    assert not measure.knows_party("Maria Lopez: Yes, I'm aware of Valentine's Day. It's a special day to celebrate love.")  # not the party
    assert measure.knows_party("Sam Moore: Yes, Isabella, there is a Valentine's Day party at Hobbs Cafe.")
    assert not measure.knows_mayor("Eddy Lin: I'm sorry, but as an AI, I don't have the ability to know who is running for mayor.")
    assert not measure.knows_person("I'm sorry, but I don't have any information about Maria Lopez.")
    assert measure.strip_name_prefix("Tom Moreno: No, I don't know.") == "No, I don't know."


def test_scripted_three_agent_simulation_spreads_the_party_by_dialogue_only():
    """Isabella talks to Maria (same place); Maria did not know about the party before. Klaus never meets anyone and
    must not know. Relationship density rises only through mutual acquaintance in memory."""
    def plan(t):
        name = re.search(r"Name: (.+?) \(", t).group(1)
        return " go to Hobbs Cafe at 7:00 am\n2) stay at the cafe until 11 pm" if name != "Klaus Mueller" else " stay in the dorm room at 7:00 am\n2) sleep in the dorm room at 11 pm"
    def hourly(t):
        return "07:00 - go to Hobbs Cafe\n09:00 - stay at the cafe" if "Klaus" not in t else "07:00 - stay in the dorm room\n09:00 - keep reading in the dorm room"
    def say(t):
        me = re.search(r"\n(.+?) and (.+?) are talking", t)
        speaker = me.group(1)
        if speaker == "Isabella Rodriguez":
            return "Hi Maria! I'm hosting a Valentine's Day party at Hobbs Cafe tomorrow at 5pm, please come!"
        return "That sounds wonderful, Isabella, I'll be there! [END]"
    def interview(t):
        q = re.search(r"Interviewer: (.+)\n", t).group(1)
        name = re.search(r"Name: (.+?) \(", t).group(1)
        if "party" in q:
            return "Yes, Isabella invited me to a Valentine's Day party at Hobbs Cafe." if name != "Klaus Mueller" else "No, I did not know there was a party."
        if "mayor" in q:
            return "I'm not sure who is running."
        if "Do you know of" in q:
            other = re.search(r"Do you know of (.+?)\?", q).group(1)
            return f"Yes, I know {other}." if name != "Klaus Mueller" and other != "Klaus Mueller" else "No, I don't know them."
        return "..."
    llm = FakeLLM([
        (lambda t: "poignancy" in t, "4"),
        (lambda t: "plan today in broad strokes" in t, plan),
        (lambda t: "Break the plan into hourly" in t, hourly),
        (lambda t: "Which single place" in t and "cafe" in re.search(r"is going to: (.*)\.", t).group(1).lower(), "Hobbs Cafe: cafe"),
        (lambda t: "Which single place" in t, "none of these"),
        (lambda t: "In two sentences, summarize" in t, "They are friends."),
        (lambda t: "Should Isabella Rodriguez react" in t, "Yes, Isabella should talk to Maria and invite her to the party."),
        (lambda t: "Should" in t, "No, there is no need to react."),
        (lambda t: "are talking" in t, say),
        (lambda t: "Interviewer:" in t, interview),
        (lambda t: "salient high-level questions" in t, "1. What is going on?"),
        (lambda t: "high-level insights" in t, "1. People are busy (because of 1)"),
    ])
    sim = Simulation(load_personas(["Isabella Rodriguez", "Maria Lopez", "Klaus Mueller"]), llm, EMB, step_minutes=60, seed=1)
    sim.run(days=1)
    assert sim.dialogues and sim.dialogues[0]["a"] == "Isabella Rodriguez" and sim.dialogues[0]["b"] == "Maria Lopez"
    maria, klaus = sim.by_name["Maria Lopez"], sim.by_name["Klaus Mueller"]
    assert any(m.kind == "dialogue" and "Valentine" in m.text for m in maria.memory.nodes)
    assert not any("Valentine" in m.text for m in klaus.memory.nodes)
    now, text = sim.clock.now, sim.clock.text()
    d = measure.diffusion(sim.agents, now, text, {"party": "Isabella Rodriguez", "mayor": "Sam Moore"})
    assert d["party"]["verified"] == 2 and d["party"]["hallucinated"] == 0 and d["mayor"]["verified"] == 0
    assert measure.invited(sim.agents) == ["Maria Lopez"]
    g = measure.relationship_graph(sim.agents, now, text, use_llm=True)
    assert g["edges"] == [("Isabella Rodriguez", "Maria Lopez")] and abs(g["density"] - 1 / 3) < 1e-9 and g["hallucinated"] == 0
    assert sim.world.location["Klaus Mueller"] == klaus.home


def test_workplace_inference_and_place_choice_order():
    from papers.generative_agents.agent import infer_workplace
    isa, maria, klaus, tom = load_personas(["Isabella Rodriguez", "Maria Lopez", "Klaus Mueller", "Tom Moreno"])
    assert infer_workplace(isa) == "Hobbs Cafe: cafe" and infer_workplace(tom) == "The Willows Market and Pharmacy: store"
    assert infer_workplace(klaus) == "Oak Hill College: library" and infer_workplace(maria) in ("Hobbs Cafe: cafe", "Oak Hill College: library")
    llm = FakeLLM([(lambda t: "Which single place" in t, "The Willows Market and Pharmacy: store"), (lambda t: "poignancy" in t, "2")])
    a = Agent(isa, llm, EMB)
    assert a.choose_place("wake up and complete the morning routine") == a.home and llm.calls == 0   # never asks for routine activities
    assert a.choose_place("open Hobbs Cafe and serve customers") == "Hobbs Cafe: cafe" and llm.calls == 0
    assert a.choose_place("buy decorations") == "The Willows Market and Pharmacy: store" and llm.calls == 1  # exact listed answer accepted
    assert a.choose_place("walk to Johnson Park") == "Johnson Park: park" and llm.calls == 1                   # named place wins
    llm2 = FakeLLM([(lambda t: "Which single place" in t, "Somewhere else entirely")])
    assert Agent(isa, llm2, EMB).choose_place("visit the pharmacy for supplies") == "The Willows Market and Pharmacy: store"  # keyword fallback
    assert Agent(isa, llm2, EMB).choose_place("do her taxes") == "Hobbs Cafe: cafe"   # unknown -> workplace when there is one


def test_anchor_times_keep_broad_plan_order():
    t = Agent.anchor_times(["wake up at 7:00 am", "work at the cafe", "lunch at 12:30 pm", "close at 10 pm", "sleep"], dt.time(7, 0))
    assert [x[0] for x in t] == [(7, 0), (9, 0), (12, 30), (22, 0), (23, 0)]
    assert Agent.anchor_times(["no times anywhere", "none here"], dt.time(8, 0))[0][0] == (8, 0)


def test_dialogue_ends_on_verbatim_repeat_and_is_stored_for_both():
    lines = iter(["Hi Klaus, how is the paper?", "Going well, Maria.", "Hi Klaus, how is the paper?"])
    llm = FakeLLM([(lambda t: "are talking" in t, lambda t: next(lines)), (lambda t: "poignancy" in t, "4")])
    sim = Simulation(load_personas(["Maria Lopez", "Klaus Mueller"]), llm, EMB, step_minutes=60)
    maria, klaus = sim.agents
    hist = sim.converse(maria, klaus, "Oak Hill College: library", T0, "t")
    assert [u for _, u in hist] == ["Hi Klaus, how is the paper?", "Going well, Maria."]   # the repeat ended it
    assert sum(m.kind == "dialogue" for m in maria.memory.nodes) == 1 and sum(m.kind == "dialogue" for m in klaus.memory.nodes) == 1
    assert maria.stats["dialogues"] == 1 and klaus.stats["dialogues"] == 1 and len(sim.dialogues) == 1

