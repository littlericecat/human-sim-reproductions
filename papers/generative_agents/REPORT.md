# Reproduction report — Generative Agents: Interactive Simulacra of Human Behavior

Paper: Park, O'Brien, Cai, Morris, Liang, Bernstein. UIST 2023, arXiv:2304.03442. Official code: joonspk-research/generative_agents (Apache-2.0).
Reproduction date: 2026-10-28. Branch/commit: see git log. Status: **local-model method reproduction, not a faithful rerun.**

## What this is and is not

The paper's agents run on OpenAI models (gpt-3.5-turbo for generation, OpenAI embeddings for retrieval), 25 agents for two game days, and its believability study uses 100 paid human raters. A faithful rerun therefore costs money and was **not** done (founder rule: $0 unless approved). What this folder contains is:

1. A from-the-paper implementation of the architecture — memory stream with recency/importance/relevance retrieval (§4.1), reflection trees (§4.2), top-down daily planning and reacting (§4.3), dialogue (§4.3.2), a text sandbox of Smallville built from the official location tables and the official 25 personas (§5), and the end-to-end measurements of §7.1 — with the language model swapped for a **local open-weight model: Qwen2.5-1.5B-Instruct (Q4_K_M, llama.cpp, CPU)** and relevance embeddings from **all-MiniLM-L6-v2**. This is the "local-model adapter" and every number below is labelled as such.
2. A recorded run (`results/ga-10x2/`, cassette `cassettes/ga-10x2.jsonl`, 1,086 unique model calls, 3.83 h on two CPU cores) that is **replayed byte-for-byte by the integration test** (`tests/test_generative_agents_replay.py`): the measurements in `results.json` are recomputed from the cassette without a model.

Scale actually run (honest): **10 agents** (Isabella Rodriguez, Maria Lopez, Klaus Mueller, Sam Moore, Jennifer Moore, Tom Moreno, Jane Moreno, John Lin, Eddy Lin, Mei Lin), **2 game days** (Feb 13–14, 2023; 07:00–23:00), **hourly steps** (paper: 25 agents, two days, the sandbox advances in minutes). The paper's seeds are present in the official persona files: Isabella's party intent, Sam's candidacy, Maria's crush on Klaus.

## What was run

`python -m papers.generative_agents.run --n 10 --days 2 --step-minutes 60 --seed 0 --out runs/ga-10x2 --cassette cassettes/ga-10x2.jsonl`

Per step every agent follows its plan (moves to the planned place), perceives the agents at that place (observations with model-rated importance 1–10), decides via the paper's §4.3.1 prompt whether to react, converses (≤4 utterances, ended early on a verbatim repeat), and reflects when the summed importance of new events exceeds 150. At the start and the end, every agent is interviewed ("Do you know of <name>?" for the network; "Did you know there is a Valentine's Day party?", "Do you know who is running for mayor?" for diffusion). Labels follow the paper's coding (yes/no), affirmative answers are verified against the memory stream as in §7.1.1, and party attendance is who is at Hobbs Cafe at 5 pm on Feb 14.

Model-side deviations, all forced by the 1.5B model and stated here rather than hidden:
- **Hourly/15-minute recursive plan decomposition is off** (`decompose=False`). With it on, the model produced incoherent decompositions ("eat dinner" at 08:30, "go to bed" at 09:00). Plans therefore use the 5–8 broad daily strokes with their stated times; a location is chosen per item (an explicitly named place > home for routine items > the persona's inferred workplace > the model's pick when it names a listed place exactly > keyword fallback).
- Temperature 0.7 (fixed seed) for plans, reactions and dialogue; 0.0 for ratings, reflections, interviews.
- Relevance uses MiniLM cosine instead of OpenAI embeddings; everything else in the retrieval score is the paper's (0.995^hours decay, min–max scaling, weights 1).

## Results vs the paper

| Measure (paper §7.1.2, 25 agents, 2 days) | Paper | This run (10 agents, 2 days, local 1.5B) |
|---|---|---|
| Agents knowing Sam's mayoral candidacy at the end | 1 → 8 of 25 (4% → 32%), none hallucinated | **1 → 0 of 10**: no agent, including Sam, named the candidate. Sam said "It's Isabella Rodriguez" (wrong; coded no). 5 dialogues mentioned "mayor/election" but none carried the fact to a listener who could restate it. |
| Agents knowing Isabella's party at the end | 1 → 13 of 25 (4% → 52%), none hallucinated | **1 → 3 of 10 (30%)**, all 3 verified against the memory stream (Isabella; Sam — invited in a dialogue on Feb 13 15:00; Jane — invited on Feb 14 20:00, i.e. after the party); 0 hallucinated under the final coding. |
| Relationship network density (mutual "Do you know of") | 0.167 → 0.74 | **0.067 → 0.289** (3 → 13 mutual edges of 45). |
| Hallucinated acquaintance answers | 1.3% (6 of 453) | **8.3% (15 of 180)**: "yes" answers with no trace of the person in the agent's memory. |
| Party coordination | 5 of 12 invited attended | **1 of 2 invited attended** (Sam; Jane was invited after the party). |
| Reflections | ~2–3 per agent per day | 9 of 10 agents reflected exactly once in two days (133 reflection statements); only 8 of 133 cite evidence in the paper's "(because of 1, 5, 3)" format — the model mostly ignored the citation format. |
| Dialogues | not reported | 38 conversations; 56 reactions out of all co-located perceptions. |

Conditions to read these numbers: a 1.5B model, 10 agents, hourly steps and no plan decomposition. The direction of every effect the paper reports is present (the party spread through dialogue only, the network densified through actual encounters, reflections synthesised plausible self-statements such as "Isabella Rodriguez is involved in the planning process for the Valentine's Day party at Hobbs Cafe"), but every magnitude is far below the paper's, and the candidacy did not spread at all. We do not claim to confirm or refute the paper's numbers: the model is not the paper's model.

## What could not be reproduced and why

- **§6 controlled evaluation (TrueSkill μ: full 29.89, no-reflection 26.88, no-reflection/no-planning 25.64, human 22.95, none 21.21; Cohen's d = 8.16; Kruskal–Wallis H(4) = 150.29).** Requires 100 paid human evaluators ranking interview responses. Not run ($0 rule). No model-as-judge substitute was run because it would not be the paper's measure.
- **The paper's model.** gpt-3.5-turbo and OpenAI embeddings need an API key; a faithful rerun needs separate founder approval. The local 1.5B model is weak at exactly the behaviours the architecture leans on: initiating conversations about its own goals (Isabella's "currently" says she is telling everyone about the party, yet most reactions were "no need to react"), answering "Do you know of X?" from retrieved memory (73 of 90 start-of-run answers were refusals such as "I don't have any information about Maria Lopez" even for seeded family members), and citing evidence in reflections.
- **Scale.** 25 agents and minute-level steps were not run: at ~1 call/s on this CPU the 10-agent hourly run already took 3.8 h; 25 agents at 30-minute steps would take several days.
- **Interview coding.** The paper coded answers by hand; here a rule-based coder is used (`measure.py`), unit-tested on the exact answers this run produced, and every "yes" is verified against the memory stream as the paper did. Raw answers are in `results.json` so the coding can be audited.

## Files

- `memory.py`, `agent.py`, `world.py`, `sim.py`, `measure.py`, `prompts.py`, `run.py` — the implementation (prompts that the paper prints are kept verbatim).
- `data/` — official personas and location tables (Apache-2.0, attributed in `data/README.md`).
- `results/ga-10x2/` — `results.json` (all measurements, interview answers, graph edges), `dialogues.json`, `events.jsonl` (plans, reactions, reflections), `agents.json` (every memory stream), `log.txt`.
- `cassettes/ga-10x2.jsonl` — every model call and response of the run.
- Tests: `tests/test_generative_agents.py` (19 unit tests with a fake model: retrieval maths, parsers, reflection threshold and citations, planning, place choice, reacting, dialogue storage, interview coding, a scripted 3-agent diffusion simulation), `tests/test_generative_agents_replay.py` (integration: the committed measurements are reproduced by replaying the cassette).
