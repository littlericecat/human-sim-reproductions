# Generative Agent Simulations of 1,000 People — data-access documentation (no reproduction attempted)

Paper: Park, Zou, Shaw, Hill, Cai, Morris, Willer, Liang, Bernstein. arXiv:2411.10109 — **v1 (2024-11-15) "Generative Agent Simulations of 1,000 People"; latest version (v3, 2026) retitled "LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals".** Same 1,052-participant study; the headline changed between versions (below).
Date of this survey: 2026-10-29. Scope, per the founder's instruction: document what is openly available, what access tiers exist and what an application requires — **then stop**. Nothing restricted was requested, downloaded or redistributed; the OSF archive listed below was *listed by name only*, not downloaded.

## 1. What the headline claim rests on

- v1 abstract: agents "replicate participants' responses on the General Social Survey **85%** as accurately as participants replicate their own answers two weeks later" (normalized accuracy 0.85 = raw 68.85% ÷ self-replication 81.25%); Big Five normalized correlation 0.80 (r 0.78 ÷ 0.95); economic games 0.66 (0.66 ÷ 0.99). Interview-based agents beat demographic- and persona-based agents by 14–15 normalized points.
- v3 abstract: on held-out GSS items, interview-only / survey-only / combined agents reach **83% / 82% / 86%** of participants' own test–retest consistency, versus 74% for demographics-only agents.
- All of these are **individual-level** quantities: each agent's answers are compared with *that participant's* own Wave-1 and Wave-2 answers. They cannot be re-checked from aggregated data. Re-checking them needs (a) the per-participant interview transcripts or the built agents and (b) the per-participant survey answers — both in the restricted tier.

## 2. What is openly available today (verified 2026-10-29)

| Resource | Where | Licence / terms | What it contains | Relevance to the 85% claim |
|---|---|---|---|---|
| Code to build and query generative agents (`genagents`) | github.com/StanfordHCI/genagents (also served at github.com/joonspk-research/genagents and …/generative_agent, same repository; last push 2024-11-18) | MIT | `GenerativeAgent` class (scratchpad + memory stream + reflection), categorical / numerical / open-ended response methods, all prompt templates. Needs an OpenAI key (`gpt-4o-mini` default). | The architecture, not the data. |
| "Demographic agent bank" | same repo, `agent_bank/populations/gss_agents/` — **3,505 agent folders** | MIT | Agents seeded from GSS demographic attributes with **fictional names and addresses**; README: "a starting point to explore the codebase". | A *demographic-based* baseline population, not the 1,052 interview agents. |
| One interview-based sample agent | same repo, `agent_bank/populations/single_agent/` (one author's own interview: `scratch.json`, `memory_stream/nodes.json` 89 KB, `embeddings.json` 4 MB) | MIT | Illustrates the structure of an interview-derived agent. | n = 1, the author; no survey answers to score against. |
| Replication materials (v3) | OSF project t6g7k, public, created 2026-04-21: a single file `replication_instructions.rar` (3.4 MB) | OSF default (not stated in the node metadata; check inside) | Unknown until opened — the v3 text says the OSF repository holds "replication files" and "the code for generating all the different agents"; v3 also states "we provide open access to aggregated responses to the fixed survey instruments used to build the survey-based agents (e.g., the GSS and Big Five Inventory)". | Possibly aggregated responses; **not** downloaded here pending the founder's decision. |
| Pre-registration | osf.io/mexkf (view-only link printed in v3) | — | Pre-registered analysis plan. | Design, not data. |
| Public survey instruments | GSS (NORC, public download), Big Five Inventory, the economic games and the five replicated experiments (cited from a large-scale replication effort) | public | The *questions*; not these participants' answers. | Lets one build the fixed-task battery, not score it. |
| Ph.D. thesis "Generative agent simulations of human behavior" (Park, Stanford, 2025) | purl.stanford.edu/jm164ch6237 | CC BY-NC 3.0 | Text; no data files listed. | Narrative only. |

**Not available anywhere public:** the 1,052 interview transcripts; the 1,052 built agents; the participants' individual survey/experiment answers (Wave 1 and Wave 2); the Bovitz recruitment attributes. The v1 SM is explicit: "instead of providing the actual interviews, we will release a sample of the interview data to illustrate its structure and format. Access to the full agent behaviors will be offered via an API."

## 3. Access tiers as the authors describe them

Source: v1 Supplementary Material "Research Access for the Agent Bank" / "Strategic Framework for Access" / "Plan and Desiderata"; the genagents README "Agent Bank Access"; the Stanford HAI policy brief *Simulating Human Behavior with AI Agents* (May 2025); the HAI news piece (quotes from the first author).

1. **Open access to aggregated responses on fixed tasks.** "Academic researchers who sign our usage agreement will be granted access to aggregated data from fixed tasks, such as the surveys and experiments conducted in our study. Researchers can query subpopulations of the Agent Bank and receive responses in the form of aggregate data." Researchers may "submit suggestions for new queries or discrete response options, which will be reviewed on a rolling basis." The README (Nov 2024) phrases this as a plan: "we plan to make aggregated responses on fixed tasks accessible for general research use in the coming months." v3 (2026) says this open access "is provided" and points to the OSF repository.
2. **Restricted access to individualized responses on open tasks.** "Researchers will need to provide a clear statement of their project's research purpose and its potential harms to the participants in the Agent Bank. Once approved, they may request access from us, and upon approval, will be granted **API access** to query the agents with custom questions." Users "will receive individualized agent responses." The README: "request restricted access by contacting the authors and following a review process that ensures ethical considerations are met."
3. **Safeguards attached to both tiers:** "usage audits, participant withdrawal options, and non-commercial use agreements, modeled after genome banks and AI model deployments"; "we will maintain an audit log of how these APIs are used and continue to monitor their usage"; access may be expanded or "restrict[ed] further if necessary." Participants may withdraw consent at any time; removal requests honoured for 25 years. The HAI brief restates: "controlled, research-only API access to our agent bank … open access to aggregated responses on fixed tasks … restrict access to individual responses on open tasks for researchers following a review process"; the news piece: researchers "must apply for access and provide strict assurances of privacy protection."
4. **What the authors deliberately did not decide in print:** whether institutional IRB approval will be required of applicants (the SM discusses this as an open axis: "we could implement increasingly stringent access controls, potentially requiring researchers to obtain IRB approval from their institution"), the review criteria, turnaround, and whether non-academic entities (a company such as Mimese) are eligible at all. The usage agreement is **non-commercial**.

## 4. What an application would require (as far as the public record states)

- A named academic research purpose and a written statement of potential harms to participants (restricted tier).
- Signing the usage agreement: non-commercial use, usage auditing, acceptance of participant withdrawal.
- Contact is "the authors" (corresponding author listed on the paper); no form, portal or email address for the agent bank is published in the paper, the README, the OSF node, or the HAI materials as of this check. The API itself is described as planned/controlled; whether it is live is not stated anywhere public we found.
- Likely but unstated: institutional affiliation and possibly IRB approval (see §3.4).

## 5. What a $0, public-data-only effort can and cannot do here

- Can: run the MIT code with a local model on the 3,505 fictional GSS-demographic agents (a demographic-based baseline, which the paper reports at 74% of test–retest, v3) — but with no participant answers to score against, there is no accuracy to compute; only internal consistency and the paper's prompts can be exercised.
- Can, if the founder allows opening the OSF archive: inspect the "aggregated responses to the fixed survey instruments" and compare aggregate distributions with public GSS marginals — a population-level check, which the paper itself frames as the open tier's purpose. This would not test the individual-level headline.
- Cannot: re-check 85% / 83–86% normalized accuracy, Big Five 0.80 or economic games 0.66 — individual-level, restricted tier only. Cannot build interview agents without interviews (the only public interview agent is one author).
- Any faithful rerun of the published pipeline also needs an OpenAI key (the code defaults to gpt-4o-mini; the paper used GPT-4o), i.e. it is not $0.

## 6. Decision needed from the founder (this task stops here)

1. Open the OSF `replication_instructions.rar` (public, 3.4 MB) to see whether the aggregated open-tier data is inside, and if so run the population-level comparison? (No application, no contact required; still "download", so it waits for the go-ahead.)
2. Apply for the restricted tier? Requires contacting the authors, an academic non-commercial purpose statement and the usage agreement — conflicts with the standing "do not contact authors" rule and with Mimese's commercial setting; not recommended without a deliberate decision.
3. Otherwise: leave this paper documented as not reproducible from public data, with the demographic-baseline exercise (§5, first bullet) as the only $0 option.

## Sources (all public, retrieved 2026-10-29)

arXiv:2411.10109 v1 PDF (SM §"Research Access for the Agent Bank", "Overview of the Agent Bank", "Strategic Framework for Access", "Plan and Desiderata"); arXiv:2411.10109 v3 (abstract; "Replication materials"; open-access statement); github.com/StanfordHCI/genagents README ("Overview", "Sample Agent", "Agent Bank Access") and tree; api.osf.io node t6g7k (public, files listed by name only); Stanford HAI policy brief "Simulating Human Behavior with AI Agents" (May 2025); Stanford HAI news "AI agents simulate 1,052 individuals' personalities with impressive accuracy"; purl.stanford.edu/jm164ch6237.
