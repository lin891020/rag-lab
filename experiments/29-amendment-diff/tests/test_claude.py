"""27's agent loop, copied with its tests: searches go out as strict tool calls and come back as
tool results matched by id, the whole assistant turn is sent back, the sixth search is refused,
a question that never stops searching is stopped at the turn cap, every turn is billed (cache
writes and reads too). Plus 29's `chat`: one turn, no tools key sent, billed the same way, a
non-end_turn stop raises with the bill taken."""

import os
from types import SimpleNamespace

import claude
import pytest

OPUS, HAIKU = "claude-opus-5-5", "claude-haiku-5-5"
TOOL = {"name": "search_old", "description": "d", "input_schema": {"type": "object"}}
ANSWER = '{"found": true, "quote": "", "expression": "8", "value": 8, "unit": "mm²"}'


def test_price_counts_cache_writes_and_reads():
    u = {"input": 1_000_000, "cache_write": 1_000_000, "cache_read": 1_000_000, "output": 1_000_000}
    assert claude.price(OPUS, u) == pytest.approx(4 + 5 + 0.20 + 20)
    assert claude.price(HAIKU, u) == pytest.approx(0.10 + 0.125 + 0.01 + 0.50)
    assert claude.price(OPUS, {"input": 1_000_000, "output": 0}) == 4.0  # 26's rows: no cache keys


def _use(i, q="表"):
    return SimpleNamespace(type="tool_use", id=f"tu_{i}", name="search_old", input={"query": q})


def _text(t=ANSWER):
    return SimpleNamespace(type="text", text=t)


class _Fake:
    """Replies in order; each turn bills 100 in, 10 cache write, 50 cache read, 7 out."""

    def __init__(self, turns):
        self.turns, self.sent = list(turns), []
        self.messages = self

    def create(self, **kw):
        self.sent.append({**kw, "messages": list(kw["messages"])})
        stop, content = self.turns.pop(0)
        return SimpleNamespace(
            stop_reason=stop,
            content=content,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=7,
                cache_creation_input_tokens=10,
                cache_read_input_tokens=50,
            ),
            model=kw["model"],
            _request_id=f"req_{len(self.sent)}",
        )


def _run(monkeypatch, turns, **kw):
    fake = _Fake(turns)
    monkeypatch.setattr(claude, "_client", lambda: fake)
    asked: list[str] = []
    usage = claude.zero_usage()

    def tool(q):
        asked.append(q)
        return f"結果 {q}"

    def go():
        return claude.agent(
            "Q",
            model=OPUS,
            system="S",
            schema={"type": "object"},
            tool=TOOL,
            run_tool=tool,
            usage=usage,
            **kw,
        )

    return fake, asked, usage, go


def test_two_searches_then_an_answer(monkeypatch):
    turns = [("tool_use", [_use(1, "甲")]), ("tool_use", [_use(2, "乙")]), ("end_turn", [_text()])]
    fake, asked, usage, go = _run(monkeypatch, turns)
    r = go()
    assert asked == ["甲", "乙"] and r["content"] == ANSWER
    assert (r["turns"], r["searches"], r["refused"]) == (3, 2, 0)
    assert usage == {"input": 300, "cache_write": 30, "cache_read": 150, "output": 21, "calls": 3}
    last = fake.sent[-1]["messages"]
    assert [m["role"] for m in last] == ["user", "assistant", "user", "assistant", "user"]
    assert last[1]["content"] == turns[0][1]  # the whole assistant turn goes back
    assert last[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "結果 甲"}
    ]
    sent = fake.sent[0]
    assert sent["tools"] == [TOOL] and sent["cache_control"] == {"type": "ephemeral"}
    assert sent["output_config"]["effort"] == claude.EFFORT and "temperature" not in sent
    assert "tool_choice" not in sent  # Opus 5.5 rejects forced tool use; auto is the default
    assert r["request_ids"] == ["req_1", "req_2", "req_3"] and r["served_model"] == OPUS


def test_parallel_searches_answer_in_one_message_in_order(monkeypatch):
    turns = [("tool_use", [_text("想"), _use(1, "甲"), _use(2, "乙")]), ("end_turn", [_text()])]
    fake, asked, _, go = _run(monkeypatch, turns)
    go()
    results = fake.sent[-1]["messages"][2]["content"]
    assert [x["tool_use_id"] for x in results] == ["tu_1", "tu_2"] and asked == ["甲", "乙"]


def test_the_search_after_the_cap_is_refused(monkeypatch):
    turns = [("tool_use", [_use(i)]) for i in range(1, 8)] + [("end_turn", [_text()])]
    fake, asked, _, go = _run(monkeypatch, turns, max_turns=10)
    r = go()
    assert len(asked) == claude.MAX_SEARCHES == 5 and (r["searches"], r["refused"]) == (5, 2)
    refused = fake.sent[-1]["messages"][-1]["content"][0]
    assert refused["is_error"] is True and refused["content"] == claude.LIMIT_MSG


def test_blank_query_is_an_error_not_a_search(monkeypatch):
    turns = [("tool_use", [_use(1, "  ")]), ("end_turn", [_text()])]
    fake, asked, _, go = _run(monkeypatch, turns)
    r = go()
    assert asked == [] and r["searches"] == 0
    assert fake.sent[-1]["messages"][-1]["content"][0]["is_error"] is True


def test_turn_cap_stops_the_question_and_keeps_the_bill(monkeypatch):
    turns = [("tool_use", [_use(i)]) for i in range(1, 9)]
    _, asked, usage, go = _run(monkeypatch, turns, max_searches=10)
    with pytest.raises(claude.StopError) as e:
        go()
    assert e.value.reason == "max_turns" and e.value.turns == claude.MAX_TURNS == 8
    assert usage["calls"] == 8 and usage["input"] == 800
    assert len(asked) == 7  # the last turn's search is not run: the model would never see it
    assert (e.value.searches, e.value.refused, e.value.served_model) == (7, 0, OPUS)
    assert len(e.value.request_ids) == 8


def test_a_refusal_is_a_stop(monkeypatch):
    _, _, usage, go = _run(monkeypatch, [("tool_use", [_use(1)]), ("refusal", [])])
    with pytest.raises(claude.StopError) as e:
        go()
    assert e.value.reason == "refusal" and usage["calls"] == 2
    assert e.value.searches == 1 and e.value.request_ids == ["req_1", "req_2"]


def test_api_schema_closes_objects_and_drops_unsupported_keys():
    s = claude.api_schema({"type": "object", "properties": {"v": {"type": "number", "minimum": 0}}})
    assert s["additionalProperties"] is False and "minimum" not in s["properties"]["v"]


def test_load_env_keeps_existing_variables_and_skips_comments(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\nA_KEY='x'\nB_KEY=y\n")
    monkeypatch.setenv("B_KEY", "kept")
    monkeypatch.delenv("A_KEY", raising=False)
    claude.load_env(p)
    assert os.environ["A_KEY"] == "x" and os.environ["B_KEY"] == "kept"


def test_chat_sends_no_tools_bills_one_turn_and_returns_the_text(monkeypatch):
    fake = _Fake([("end_turn", [_text('{"found": true}')])])
    monkeypatch.setattr(claude, "_client", lambda: fake)
    usage = claude.zero_usage()
    r = claude.chat("Q", model=HAIKU, system="S", schema={"type": "object"}, usage=usage)
    assert r["content"] == '{"found": true}' and r["turns"] == 1 and r["searches"] == 0
    sent = fake.sent[0]
    assert "tools" not in sent and sent["messages"] == [{"role": "user", "content": "Q"}]
    assert sent["output_config"]["effort"] == claude.EFFORT
    assert usage == {"input": 100, "cache_write": 10, "cache_read": 50, "output": 7, "calls": 1}
    assert r["request_ids"] == ["req_1"] and r["served_model"] == HAIKU


def test_chat_stop_other_than_end_turn_raises_with_the_bill_taken(monkeypatch):
    fake = _Fake([("max_tokens", [_text("partial")])])
    monkeypatch.setattr(claude, "_client", lambda: fake)
    usage = claude.zero_usage()
    with pytest.raises(claude.StopError) as e:
        claude.chat("Q", model=OPUS, system="S", schema={"type": "object"}, usage=usage)
    assert (
        e.value.reason == "max_tokens" and usage["calls"] == 1 and e.value.request_ids == ["req_1"]
    )
