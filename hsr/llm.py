"""Local language model with a recording cassette.

`LocalLLM` runs an open-weight GGUF through llama.cpp on the CPU ($0). Every call is keyed by a hash of its
inputs and, when a cassette is attached, the response is recorded to it; in *replay* mode the cassette is the
only source and a missing entry is an error — that is how the tests run without a model and how a report's
numbers are reproduced byte for byte. `FakeLLM` answers from a rule table for unit tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_MODEL = os.environ.get("HSR_MODEL", str(Path.home() / ".cache/hs-models/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
DEFAULT_EMBED = os.environ.get("HSR_EMBED", str(Path.home() / ".cache/hs-models/all-MiniLM-L6-v2-Q8_0.gguf"))

Message = dict[str, str]


def call_key(messages: list[Message], max_tokens: int, temperature: float, seed: Optional[int], model_tag: str) -> str:
    payload = json.dumps({"m": messages, "t": max_tokens, "T": temperature, "s": seed, "model": model_tag}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class Cassette:
    """JSON-lines store of {key, messages, response, usage, model}. Thread-safe append; whole-file load."""

    def __init__(self, path: str | Path, mode: str = "record") -> None:
        assert mode in ("record", "replay", "off")
        self.path, self.mode = Path(path), mode
        self.entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    e = json.loads(line)
                    self.entries[e["key"]] = e

    def get(self, key: str) -> Optional[dict[str, Any]]:
        return self.entries.get(key)

    def put(self, key: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self.entries[key] = entry
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, **entry}, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self.entries)


class LLM:
    """Interface: chat(messages, ...) -> str; next_token_logprobs(prompt, k) -> {token: logprob} for the next token."""

    model_tag = "abstract"
    calls = 0

    def chat(self, messages: list[Message], max_tokens: int = 128, temperature: float = 0.0, seed: Optional[int] = 0) -> str:
        raise NotImplementedError

    def complete(self, prompt: str, system: Optional[str] = None, **kw: Any) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(msgs, **kw)

    def next_token_logprobs(self, prompt: str, k: int = 60) -> dict[str, float]:
        raise NotImplementedError

    def complete_raw(self, prompt: str, max_tokens: int = 5, temperature: float = 0.7, seed: Optional[int] = 0) -> str:
        """Plain (non-chat) continuation of a prompt, as the GPT-3 completion API would do."""
        raise NotImplementedError


def logprob_key(prompt: str, k: int, model_tag: str) -> str:
    return hashlib.sha256(json.dumps({"lp": prompt, "k": k, "model": model_tag}, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


def raw_key(prompt: str, max_tokens: int, temperature: float, seed: Optional[int], model_tag: str) -> str:
    return hashlib.sha256(json.dumps({"raw": prompt, "t": max_tokens, "T": temperature, "s": seed, "model": model_tag}, ensure_ascii=False).encode("utf-8")).hexdigest()[:24]


class LocalLLM(LLM):
    def __init__(self, model_path: str = DEFAULT_MODEL, cassette: Optional[Cassette] = None, n_ctx: int = 4096, n_threads: int = 2,
                 logits_all: bool = False) -> None:
        self.model_path, self.cassette, self.n_ctx, self.n_threads, self.logits_all = model_path, cassette, n_ctx, n_threads, logits_all
        self.model_tag = Path(model_path).name
        self._llm = None
        self._lock = threading.Lock()

    def _load(self):
        if self._llm is None:
            from llama_cpp import Llama  # imported lazily so replay mode needs no llama.cpp

            self._llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, verbose=False, seed=0, logits_all=self.logits_all)
        return self._llm

    def close(self) -> None:
        """Free the loaded model (e.g. to reload with different llama.cpp options); the cassette stays attached."""
        self._llm = None

    def _cached(self, key: str) -> Optional[dict[str, Any]]:
        if self.cassette is not None and self.cassette.mode != "off":
            hit = self.cassette.get(key)
            if hit is not None:
                return hit
            if self.cassette.mode == "replay":
                raise LookupError(f"cassette {self.cassette.path.name}: no recording for key {key} (replay mode)")
        return None

    def next_token_logprobs(self, prompt: str, k: int = 60) -> dict[str, float]:
        """Top-k log-probabilities of the next token after `prompt` (needs logits_all=True)."""
        key = logprob_key(prompt, k, self.model_tag)
        hit = self._cached(key)
        if hit is not None:
            return dict(hit["logprobs"])
        with self._lock:
            llm = self._load()
            out = llm.create_completion(prompt, max_tokens=1, temperature=0.0, logprobs=k)
        lp = {tok: float(v) for tok, v in out["choices"][0]["logprobs"]["top_logprobs"][0].items()}
        self.calls += 1
        if self.cassette is not None and self.cassette.mode == "record":
            self.cassette.put(key, {"kind": "logprobs", "prompt": prompt, "k": k, "logprobs": lp, "model": self.model_tag})
        return lp

    def complete_raw(self, prompt: str, max_tokens: int = 5, temperature: float = 0.7, seed: Optional[int] = 0) -> str:
        key = raw_key(prompt, max_tokens, temperature, seed, self.model_tag)
        hit = self._cached(key)
        if hit is not None:
            return hit["response"]
        with self._lock:
            llm = self._load()
            if seed is not None:
                llm.set_seed(seed)
            out = llm.create_completion(prompt, max_tokens=max_tokens, temperature=temperature)
        text = out["choices"][0]["text"]
        self.calls += 1
        if self.cassette is not None and self.cassette.mode == "record":
            self.cassette.put(key, {"kind": "raw", "prompt": prompt, "response": text, "max_tokens": max_tokens, "temperature": temperature, "seed": seed, "model": self.model_tag})
        return text

    def chat(self, messages: list[Message], max_tokens: int = 128, temperature: float = 0.0, seed: Optional[int] = 0) -> str:
        key = call_key(messages, max_tokens, temperature, seed, self.model_tag)
        if self.cassette is not None and self.cassette.mode != "off":
            hit = self.cassette.get(key)
            if hit is not None:
                return hit["response"]
            if self.cassette.mode == "replay":
                raise LookupError(f"cassette {self.cassette.path.name}: no recording for key {key} (replay mode)")
        with self._lock:
            llm = self._load()
            if seed is not None:
                llm.set_seed(seed)
            out = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=temperature)
        text = out["choices"][0]["message"]["content"]
        self.calls += 1
        if self.cassette is not None and self.cassette.mode == "record":
            self.cassette.put(key, {"messages": messages, "response": text, "usage": out.get("usage"), "model": self.model_tag,
                                    "max_tokens": max_tokens, "temperature": temperature, "seed": seed})
        return text


class ReplayLLM(LLM):
    """Answers only from a cassette (tests, report reproduction)."""

    def __init__(self, cassette: Cassette, model_tag: str) -> None:
        self.cassette, self.model_tag = cassette, model_tag

    def chat(self, messages: list[Message], max_tokens: int = 128, temperature: float = 0.0, seed: Optional[int] = 0) -> str:
        key = call_key(messages, max_tokens, temperature, seed, self.model_tag)
        hit = self.cassette.get(key)
        if hit is None:
            raise LookupError(f"cassette {self.cassette.path.name}: no recording for key {key}")
        self.calls += 1
        return hit["response"]

    def next_token_logprobs(self, prompt: str, k: int = 60) -> dict[str, float]:
        hit = self.cassette.get(logprob_key(prompt, k, self.model_tag))
        if hit is None:
            raise LookupError(f"cassette {self.cassette.path.name}: no logprob recording for this prompt")
        self.calls += 1
        return dict(hit["logprobs"])

    def complete_raw(self, prompt: str, max_tokens: int = 5, temperature: float = 0.7, seed: Optional[int] = 0) -> str:
        hit = self.cassette.get(raw_key(prompt, max_tokens, temperature, seed, self.model_tag))
        if hit is None:
            raise LookupError(f"cassette {self.cassette.path.name}: no raw recording for this prompt")
        self.calls += 1
        return hit["response"]


class FakeLLM(LLM):
    """Deterministic stand-in for unit tests: the first rule whose predicate matches the prompt text answers."""

    model_tag = "fake"

    def __init__(self, rules: Optional[list[tuple[Callable[[str], bool], Callable[[str], str] | str]]] = None, default: str = "",
                 logprob_fn: Optional[Callable[[str], dict[str, float]]] = None) -> None:
        self.rules, self.default, self.log = list(rules or []), default, []
        self.logprob_fn = logprob_fn

    def next_token_logprobs(self, prompt: str, k: int = 60) -> dict[str, float]:
        self.log.append(prompt)
        self.calls += 1
        return dict(self.logprob_fn(prompt)) if self.logprob_fn else {}

    def complete_raw(self, prompt: str, max_tokens: int = 5, temperature: float = 0.7, seed: Optional[int] = 0) -> str:
        return self.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=temperature, seed=seed)

    def chat(self, messages: list[Message], max_tokens: int = 128, temperature: float = 0.0, seed: Optional[int] = 0) -> str:
        text = "\n".join(m["content"] for m in messages)
        self.log.append(text)
        self.calls += 1
        for pred, answer in self.rules:
            if pred(text):
                return answer(text) if callable(answer) else answer
        return self.default
