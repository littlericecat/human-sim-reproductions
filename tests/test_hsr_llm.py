"""Unit tests for the shared LLM layer: cassette record/replay keying, FakeLLM rules, embedders."""

import json

import numpy as np
import pytest

from hsr import embed
from hsr.llm import Cassette, FakeLLM, LocalLLM, ReplayLLM, call_key


def test_call_key_is_stable_and_sensitive():
    m = [{"role": "user", "content": "hi"}]
    assert call_key(m, 8, 0.0, 0, "x") == call_key(m, 8, 0.0, 0, "x")
    assert call_key(m, 8, 0.0, 0, "x") != call_key(m, 9, 0.0, 0, "x")
    assert call_key(m, 8, 0.0, 0, "x") != call_key(m, 8, 0.0, 1, "x")
    assert call_key(m, 8, 0.0, 0, "x") != call_key(m, 8, 0.0, 0, "y")


def test_cassette_roundtrip_and_replay(tmp_path):
    path = tmp_path / "c.jsonl"
    c = Cassette(path, mode="record")
    c.put("k1", {"messages": [], "response": "A"})
    c.put("k2", {"messages": [], "response": "B"})
    again = Cassette(path, mode="replay")
    assert len(again) == 2 and again.get("k1")["response"] == "A" and again.get("nope") is None
    lines = path.read_text().splitlines()
    assert [json.loads(l)["key"] for l in lines] == ["k1", "k2"]


def test_replay_llm_answers_only_from_cassette(tmp_path):
    msgs = [{"role": "user", "content": "What is 2+2?"}]
    c = Cassette(tmp_path / "c.jsonl", mode="record")
    c.put(call_key(msgs, 8, 0.0, 0, "m"), {"messages": msgs, "response": "4"})
    llm = ReplayLLM(c, "m")
    assert llm.chat(msgs, max_tokens=8) == "4"
    with pytest.raises(LookupError):
        llm.chat([{"role": "user", "content": "unrecorded"}], max_tokens=8)


def test_local_llm_replay_mode_never_loads_a_model(tmp_path):
    msgs = [{"role": "user", "content": "x"}]
    c = Cassette(tmp_path / "c.jsonl", mode="replay")
    c.put(call_key(msgs, 8, 0.0, 0, "nonexistent.gguf"), {"messages": msgs, "response": "y"})
    llm = LocalLLM(model_path="/nonexistent/nonexistent.gguf", cassette=c)
    assert llm.chat(msgs, max_tokens=8) == "y" and llm._llm is None
    with pytest.raises(LookupError):
        llm.chat([{"role": "user", "content": "z"}], max_tokens=8)


def test_fake_llm_rules():
    f = FakeLLM([(lambda t: "rate" in t, "7"), (lambda t: "plan" in t, lambda t: "1) wake up")], default="ok")
    assert f.complete("rate this memory") == "7"
    assert f.complete("make a plan") == "1) wake up"
    assert f.complete("anything") == "ok" and f.calls == 3 and len(f.log) == 3


def test_bag_of_words_embedder_is_unit_norm_and_discriminative():
    e = embed.BagOfWordsEmbedder()
    a, b, c = e.embed("Klaus is writing a research paper"), e.embed("Klaus writes his research paper"), e.embed("the refrigerator is empty")
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    assert embed.cosine(a, b) > embed.cosine(a, c)
    assert e.embed("Klaus is writing a research paper") is a  # cached
