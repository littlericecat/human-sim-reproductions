# Reproduction report — Out of One, Many: Using Language Models to Simulate Human Samples

Paper: Argyle, Busby, Fulda, Gubler, Rytting, Wingate. Political Analysis 31(3), 2023; arXiv:2209.06899. Replication package: Harvard Dataverse doi:10.7910/DVN/JPV20K (CC0).
Reproduction date: 2026-10-29. Status: **method reproduction with a local open-weight model (Qwen2.5-1.5B-Instruct Q4, llama.cpp, CPU), not a GPT-3 rerun**, plus a re-analysis of GPT-3's own answers from the CC0 file. Every model call is recorded in `cassettes/ooom.jsonl` (2,175 prompts, 4.6 h) and the numbers below are regenerated from it by `tests/test_out_of_one_many_replay.py` without a model.

## Scope

- **Data.** The paper's Study 2 uses the 2012, 2016 and 2020 ANES and Study 3 the 2016 ANES. Raw ANES files require registration and are not redistributed; the CC0 package contains the 2016 ANES items for 4,270 respondents (`anesgpt3_task3.csv`) together with GPT-3's answers. **Only 2016 is reproduced; 2012 and 2020 are not** (no public copy of the conditioning variables). Study 1 (Pigeonholing Partisans free text judged by 2,873 Lucid raters) needs paid human raters and was not run.
- **Model.** GPT-3 davinci (2022) is replaced by a 1.5B local model. Argyle et al. measure *algorithmic fidelity of GPT-3*; this report measures whether their *method* produces the same kind of correspondence with a small open model — and reports distribution-level checks (Bisbee et al., *Political Analysis* 2024, doi:10.1017/pan.2024.5) as the control the founder asked for.
- **Scale actually run.** Study 2: 1,000 of the 2,609 two-party voters (seeded random sample), next-token probabilities. Study 3: 100 respondents × 12 one-item-out completions (paper: 4,270 × 12 ≈ 50k). GPT-3 re-analysis: all 4,270 (1,784 complete cases).

## What was run

`python -m papers.out_of_one_many.run --n2 1000 --n3 100 --seed 0 --out runs/ooom --cassette cassettes/ooom.jsonl`

- **Study 2.** A first-person backstory from race, gender, age, ideology, party ID, political interest, church attendance, political discussion and flag feelings (the paper's variable list; its exact fragments are not printed, ours are in `prompts.py`), ending in "In 2016, I voted for". The top-60 next-token log-probabilities are collapsed with the paper's token sets (Appendix C.1: trump/donald/republican/conservative vs clinton/hillary/rodham/senator/democrat/democratic/liberal, case-insensitive), normalised, and dichotomised at 0.50; compared with the ANES self-reported vote by tetrachoric correlation (bivariate-normal MLE, as R's psych), Cohen's κ and proportion agreement for the paper's Table 1 subgroups.
- **Study 3.** The interview template of the paper's Figure 10 / Table 11 (verbatim wording, eleven answered items, the target last), 5 sampled tokens at temperature 0.7, exact-match parsing to the allowed options after lower-casing (non-matches are missing), Cramer's V for the 55 item pairs — human (ANES × ANES) vs silicon (ANES conditioning value × generated answer), as the paper defines it.
- **Control (Bisbee et al.).** For each item: total-variation distance between the human and silicon response distributions, Shannon entropy, modal share and (ordinal items) standard deviation; the vote distribution within each party-ID group; and a logistic regression of Trump-vs-Clinton on ideology and party ID fitted to the human vote and to the silicon vote with the same covariates.

## Results

### Study 2 — vote prediction (2016; local model, n = 1,000 two-party voters)

| | Paper (GPT-3, 2016) | Local model, paper's procedure (cut at 0.50) | Local model, threshold-free / quantile-matched |
|---|---|---|---|
| Whole sample: tetrachoric / agreement | 0.92 / 0.87 | **0.72 / 0.52** (κ 0.09) | AUC 0.93, Spearman 0.74; cut at the quantile matching the human Republican share (p > 0.686): **0.90 / 0.86** (κ 0.72) |
| Share predicted Republican | ≈ human share | **0.945** (human 0.469) | 0.469 by construction |

The model's conditional ordering is right — mean P(Republican) rises monotonically from strong Democrats (0.55) to strong Republicans (0.86) and from extremely liberal (0.55) to conservative (0.83) — but its *marginal* is wrong: after any backstory the first token " Donald" dominates, so the paper's fixed 0.50 cut calls 94.5% of respondents Trump voters and the agreement collapses. The subgroup pattern of Table 1 is therefore not reproduced at the 0.50 cut (strong partisans 0.76 / 0.56 vs the paper's 1.00 / 0.97; liberals 0.37 / 0.20 vs 0.73 / 0.95; Blacks 0.08 / 0.17 vs 0.87 / 0.96; full table in `results/ooom/results.json` → `study2.subgroups`). With the quantile-matched cut the whole-sample numbers are within 0.02 of the paper's — but that cut uses the human marginal, which the paper's procedure does not. **Reading:** the "silicon sample" carries human-like *relative* vote patterns even in a 1.5B model, while the step the paper relies on (an uncalibrated model's absolute probabilities) does not transfer.

### Study 3 — closed-ended questions (2016)

| Quantity | Paper (GPT-3, main text) | GPT-3 re-analysed from the CC0 file (ours) | Local model (ours, n = 100) |
|---|---|---|---|
| Complete cases | 1,782 | 1,784 | 61 |
| Mean (ANES − silicon) Cramer's V over 55 pairs | "−0.026" (text); Appendix Tables 11–13 give +0.030 | **+0.027** | +0.030 (26 of 55 pairs defined) |
| Mean absolute difference | — (appendix: 0.058) | 0.051 | 0.136 |
| Correlation of the 55 pair values, ANES vs silicon | — (appendix implies 0.68) | **0.79** | 0.64 (26 pairs) |

Our human Cramer's V values match the appendix at r = 0.999 (mean absolute difference 0.003) once age is treated as a raw categorical variable, which is how the appendix numbers were evidently computed (with four age bands the age pairs fall from ≈0.20 to ≈0.05). Our GPT-3 silicon values match the appendix at r = 0.90 (MAD 0.022; the residual comes from 1,784 vs 1,782 complete cases and recoding details). Two observations about the paper's own analysis: the main-text mean difference (−0.026) has the opposite sign to its appendix table (+0.030, i.e. GPT-3 associations are on average *weaker* than the human ones); and "pattern correspondence" rests on a 0.68–0.79 correlation across pairs with a mean absolute gap of ≈0.05 — real, but far from the "stunning correspondence" wording.

The local model's Study 3 is a failure of the method at this scale and model size, and we report it as such: every silicon respondent answered "white" to race and "moderate" to ideology, 92% "Donald Trump" to vote choice, 85% "male"; 29 of 55 pairs are undefined because a silicon variable had a single category. Parse rate was 100% on 11 items and 92% on vote choice — the model follows the format, it just answers the same thing.

### Control — distribution match (Bisbee et al.)

| Item | TV distance, local | entropy human / local | TV distance, GPT-3 | entropy human / GPT-3 |
|---|---|---|---|---|
| gender | 0.33 | 1.00 / 0.61 | 0.27 | 1.00 / 0.83 |
| race | 0.25 | 1.14 / 0.00 | 0.22 | 1.17 / 0.21 |
| ideology | 0.75 | 2.56 / 0.00 | 0.32 | 2.59 / 2.61 |
| party ID | 0.63 | 2.77 / 1.50 | 0.28 | 2.77 / 2.45 |
| vote choice | 0.54 | 1.33 / 0.00 | 0.43 | 1.30 / 1.50 |
| church attendance | 0.26 | 0.95 / 0.47 | 0.00 | 0.97 / 0.97 |
| age (raw) | 0.82 (sd 17.2 vs 10.0) | 5.38 / 2.62 | 0.42 (sd 17.6 vs 12.4) | 6.04 / 5.33 |

Logistic regression of Trump vote on ideology and party ID (both centred): human coefficients 0.74 / 1.08 (n = 1,157, GPT-3 cases) vs GPT-3 silicon 0.36 / 0.82 — same signs, both attenuated by a third to a half; local silicon −0.01 / −0.01 with intercept 10.9 (no variation to explain). Vote distribution within party-ID groups: TV distance 0.24–0.61 for GPT-3, 0.0–0.94 for the local model.

**Reading.** Even for GPT-3's own answers the Bisbee pattern is visible in this data: marginals are compressed toward the mode (race 97% white vs 75%; age sd 12.4 vs 17.6; education modal share 0.66 vs 0.36), entropies are lower on 9 of 12 items, and regression coefficients are attenuated — while the pairwise Cramer's V correspondence that the paper reports still holds at r ≈ 0.8. Matching association patterns on average and matching distributions are different claims; the second is the one that fails first, and it fails completely for the small local model.

## What matches, what does not, what could not be reproduced

- Matches (from the CC0 data, no model): the paper's human Cramer's V table (r = 0.999); GPT-3's pattern correspondence in the sense of a 0.79 correlation across pairs and a mean difference of ≈0.03 — but with the sign of the appendix, not the text.
- Does not match (local model): Table 1 at the paper's 0.50 cut (0.72 / 0.52 vs 0.92 / 0.87); Study 3 pattern correspondence (0.64 on 26 defined pairs, with degenerate silicon marginals).
- Partially: the conditional structure (AUC 0.93; quantile-matched 0.90 / 0.86) shows the method extracts real demographic→vote associations from a 1.5B model, which is the paper's Criterion 4 in a weak form.
- Not reproducible here: Study 1 (paid human raters); 2012 and 2020 waves (ANES registration; not in the package); GPT-3 itself (deprecated API, paid). Study 3 at full scale (50k completions ≈ 8 days on this CPU) — only 100 respondents were run; at this n, 29 pairs were undefined, so the Study 3 local numbers are indicative only.

## Files

`data.py` (recoding; bins listed in the docstrings), `prompts.py` (Figure 10 template verbatim; our Study 2 fragments), `study2.py`, `study3.py`, `control.py`, `stats.py`, `run.py`; `data/` (CC0 file, paper tables, attribution); `results/ooom/` (`results.json` with every subgroup and pair, `study2_predictions.csv`, `study3_generated.csv` with raw completions, `log.txt`); `cassettes/ooom.jsonl`. Tests: `tests/test_out_of_one_many.py` (recoding, template text, token-set probabilities, parsing, tetrachoric/κ/Cramer's V/TV/entropy/logit, subgroup masks, control, and the appendix re-analysis check), `tests/test_out_of_one_many_replay.py` (results reproduced from the cassette; report quotes results).
