# human-sim-reproductions

Independent reproductions of human-simulation papers, run with **local open-weight models on a CPU** ($0 API
spend), with unit + integration tests and a written report per paper that says what was run, which of the
paper's numbers were matched, which were not, and what could not be reproduced and why.

| # | Paper | Folder | Status |
|---|---|---|---|
| 1 | Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (UIST 2023, arXiv:2304.03442) | `papers/generative_agents/` | done — local-model method reproduction (10 agents × 2 days); see REPORT.md |
| 2 | Argyle et al., *Out of One, Many: Using Language Models to Simulate Human Samples* (Political Analysis 2023, arXiv:2209.06899) | `papers/out_of_one_many/` | done — local-model method reproduction (2016 ANES; n = 1,000 / 100) + GPT-3 re-analysis; see REPORT.md |
| 3 | Park et al., *Generative Agent Simulations of 1,000 People* (arXiv:2411.10109v1) | `papers/thousand_people/` | access-tier documentation only (founder decision pending) |

## Setup (local models only)

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e ".[local,dev]" \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
mkdir -p ~/.cache/hs-models && cd ~/.cache/hs-models
curl -LO https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
curl -LO https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf
.venv/bin/pytest -q            # replays the committed cassettes: no model needed
```

Every model call goes through `hsr/llm.py` and is recorded to a cassette (`cassettes/`), so each report's
numbers can be replayed exactly; the tests run in replay mode. `HSR_MODEL` / `HSR_EMBED` override the model paths.

## Rules followed

No paid API. No contact with the papers' authors. No public claims beyond the reports in this repository.
Numbers in the reports are produced by the code in this repository or quoted from the papers with a citation.
