"""One delivery, one turn.

His report, 2026-09-08: *"you sendt me two output messages that is
different, but related the exact same information"*. Later the same morning:
a block of tool calls running under a reply he had already been sent, and a
tools drawer swapping between two runs' steps. One fault behind all three --
`/generate` received the same turn twice and ran it twice.

`_invocation_lock` in cli.py was already stopping two turns running at
*once*. It serialises; it does not dedupe. The second request waited for the
first and then ran the whole turn again, so one question was answered twice
and, because a model asked the same thing twice does not answer identically,
answered differently.
"""
import contextlib
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from bridge import server
from bridge.cli import ClaudeCliError, TurnCancelled, UsageLimitError


class _FakeHandler(server.BridgeHandler):
    """BridgeHandler without its socket. Same shape test_cancel.py uses."""

    def __init__(self, path, body):
        self.path = path
        self.headers = {"Content-Length": str(len(body))}
        self.rfile = MagicMock()
        self.rfile.read.return_value = body.encode()
        self.sent = []

    def _send(self, status, payload):
        self.sent.append((status, payload))


@pytest.fixture(autouse=True)
def _clean_registry():
    with server._dedupe_lock:
        server._dedupe.clear()
    yield
    with server._dedupe_lock:
        server._dedupe.clear()


def _deliver(payload, results):
    """Run one /generate to completion and hand back (status, payload)."""
    handler = _FakeHandler("/generate", json.dumps(payload))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "generate", side_effect=results):
        handler.do_POST()
    return handler.sent[-1]


TURN = {"conversation_id": "conv-1", "prompt": "fix the drawer"}


def _post_over_a_dead_socket(payload, results):
    """One /generate whose response cannot be written -- the caller has
    stopped waiting. This is the shape behind his duplicate replies."""
    handler = _FakeHandler("/generate", json.dumps(payload))
    handler._send = MagicMock(side_effect=BrokenPipeError(32, "Broken pipe"))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "generate", side_effect=results):
        handler.do_POST()


def test_the_same_question_after_an_answer_is_a_new_turn():
    """The boundary, and the half that keeps this safe.

    I built the memory the other way first -- finished turns remembered for
    two minutes, so a late retry could be answered from the record -- and
    seventeen existing tests failed, which was the design saying out loud
    what it should have said quietly: content is not identity. Once an
    answer exists, the same sentence arriving again is far more likely to be
    him asking on purpose (the "ask again" button, or retyping a line) than a
    retry, and swallowing that is worse than the bug being fixed."""
    calls = []

    def counted(*a, **k):
        calls.append(a)
        return (f"answer {len(calls)}", "")

    first = _deliver(TURN, counted)
    second = _deliver(TURN, counted)
    assert first == (200, {"text": "answer 1", "thinking": ""})
    assert second == (200, {"text": "answer 2", "thinking": ""}), \
        "asking the same thing again replayed the old answer"
    assert len(calls) == 2


def test_a_delivery_that_arrives_mid_turn_waits_for_it_rather_than_running():
    """The case he actually hit: the retry lands while the first turn is
    still going. It used to queue on `_invocation_lock` and then run again;
    now it blocks on the turn in flight and is handed its answer."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow(*a, **k):
        calls.append(a)
        started.set()
        release.wait(5)
        return ("the answer", "")

    out = {}

    def run(name):
        out[name] = _deliver(TURN, slow)

    first = threading.Thread(target=run, args=("first",))
    first.start()
    assert started.wait(5), "the first turn never started"

    second = threading.Thread(target=run, args=("second",))
    second.start()
    # It must be waiting, not running: give it room to get it wrong.
    time.sleep(0.2)
    assert len(calls) == 1, "the second delivery started a turn of its own"

    release.set()
    first.join(5)
    second.join(5)
    assert out["first"] == (200, {"text": "the answer", "thinking": ""})
    assert out["second"] == out["first"], "the joiner got a different answer"
    assert len(calls) == 1, f"the turn ran {len(calls)} times, not once"


def test_a_different_prompt_in_the_same_conversation_is_a_different_turn():
    """The negative control, and the one that would make this dangerous if
    it were wrong: deduping on the conversation alone would swallow his next
    message."""
    calls = []

    def counted(*a, **k):
        calls.append(a)
        return (f"answer {len(calls)}", "")

    _deliver(TURN, counted)
    second = _deliver({"conversation_id": "conv-1", "prompt": "and the other one?"},
                      counted)
    assert len(calls) == 2, "a new question was answered from the previous turn"
    assert second == (200, {"text": "answer 2", "thinking": ""})


def test_the_same_sentence_under_a_different_image_is_a_different_turn():
    """He sends screenshots with one word of text. The attachments are part
    of the question, so they are part of what makes two deliveries the same
    delivery."""
    calls = []

    def counted(*a, **k):
        calls.append(a)
        return (f"answer {len(calls)}", "")

    _deliver({"conversation_id": "conv-1", "prompt": "see image",
              "attachments": ["a.png"]}, counted)
    _deliver({"conversation_id": "conv-1", "prompt": "see image",
              "attachments": ["b.png"]}, counted)
    assert len(calls) == 2, "a second screenshot was answered from the first"


def test_a_joiner_gets_the_outcome_the_turn_actually_had():
    """Not just a success. A stop and a usage limit are both answers to the
    question that was asked, and a joiner that got a 200 with empty text
    while the real turn was stopped would render as a reply of nothing."""
    for raised, expected in [
        (TurnCancelled("half an answer", "some thinking"),
         (200, {"text": "half an answer", "thinking": "some thinking",
                "stopped": True})),
        (UsageLimitError("out of quota"), 429),
    ]:
        with server._dedupe_lock:
            server._dedupe.clear()
        started, release, calls = threading.Event(), threading.Event(), []

        def slow(*a, **k):
            calls.append(a)
            started.set()
            release.wait(5)
            raise raised

        out = {}
        first = threading.Thread(target=lambda: out.__setitem__("first", _deliver(TURN, slow)))
        first.start()
        assert started.wait(5)
        second = threading.Thread(target=lambda: out.__setitem__("second", _deliver(TURN, slow)))
        second.start()
        time.sleep(0.1)
        release.set()
        first.join(5)
        second.join(5)
        assert len(calls) == 1, "the joiner ran a turn of its own"
        if isinstance(expected, tuple):
            assert out["first"] == expected
        else:
            assert out["first"][0] == expected
        assert out["second"] == out["first"], "the joiner got a different outcome"


def test_a_turn_that_crashed_tells_its_joiner_rather_than_hanging():
    """An unhandled error is not an answer. Whoever is waiting has to be
    released with something honest -- a hang would be the worst outcome of
    the three."""
    started, release, calls = threading.Event(), threading.Event(), []

    def crash(*a, **k):
        calls.append(a)
        started.set()
        release.wait(5)
        raise RuntimeError("boom")

    out = {}
    first = threading.Thread(target=lambda: out.__setitem__("first", _deliver(TURN, crash)))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=lambda: out.__setitem__("second", _deliver(TURN, crash)))
    second.start()
    time.sleep(0.1)
    release.set()
    first.join(5)
    second.join(5)
    assert out["first"][0] == 500
    assert out["second"][0] == 503, "the joiner was left hanging or told it succeeded"
    assert len(calls) == 1


def test_a_turn_refused_by_the_drain_leaves_no_record_either():
    """503 means it never started. A record saying otherwise would make the
    retry against the replacement pod replay the refusal."""
    handler = _FakeHandler("/generate", json.dumps(TURN))
    with patch.object(server, "BRIDGE_TOKEN", ""), \
            patch.object(server, "_enter_turn", return_value=False):
        handler.do_POST()
    assert handler.sent[-1][0] == 503
    with server._dedupe_lock:
        assert server._dedupe == {}, "a refused delivery was recorded as answered"


def test_nothing_is_remembered_once_the_turn_is_over():
    """The registry is not a cache. It holds exactly the turns running right
    now, so a long-lived pod cannot accumulate one entry per question he has
    ever asked."""
    _deliver(TURN, lambda *a, **k: ("the answer", ""))
    with server._dedupe_lock:
        assert server._dedupe == {}, "a finished turn was left in the registry"


def test_an_answer_the_caller_never_received_is_kept_for_the_retry():
    """The case the in-flight window did not cover, and the one he actually
    hit. From this pod's own log, 2026-09-08:

        [12:20:46] /generate failed: [Errno 32] Broken pipe

    The turn had finished; the caller had stopped waiting, so writing the
    response failed. Agora saw nothing come back and delivered the same
    message again -- and he got two replies to one question.

    A broken pipe is not a guess about whether two prompts are "the same
    question". It is a fact about the connection: this answer was never
    received, so the delivery that follows it is a retry.
    """
    calls = []

    def once(*a, **k):
        calls.append(a)
        return ("the answer", "")

    # The 500 the outer handler then tries to send fails on the same dead
    # socket, which is why the pod's log line is the last word on that turn.
    with contextlib.suppress(BrokenPipeError):
        _post_over_a_dead_socket(TURN, once)
    assert len(calls) == 1

    # The retry Agora sends because it believes it got nothing.
    second = _deliver(TURN, once)
    assert second == (200, {"text": "the answer", "thinking": ""}), \
        "the retry was answered with something other than the turn's own answer"
    assert len(calls) == 1, "the retry ran the turn a second time"


def test_a_kept_answer_is_dropped_once_the_retry_window_passes():
    """Past the window an identical prompt is him asking again, not a
    retry -- the same line the in-flight rule draws, one step later."""
    calls = []

    def counted(*a, **k):
        calls.append(a)
        return (f"answer {len(calls)}", "")

    with contextlib.suppress(BrokenPipeError):
        _post_over_a_dead_socket(TURN, counted)

    with server._dedupe_lock:
        for entry in server._dedupe.values():
            entry["undelivered_at"] = time.time() - server.DELIVERY_RETRY_SECONDS - 1
    second = _deliver(TURN, counted)
    assert len(calls) == 2, "the same question could never be asked again"
    assert second == (200, {"text": "answer 2", "thinking": ""})


def test_a_delivered_answer_is_still_forgotten_immediately():
    """The negative control for the pair above: retention is for the failed
    send and nothing else. A delivered answer must not be replayed, or
    tapping "ask again" would hand him the previous reply."""
    calls = []

    def counted(*a, **k):
        calls.append(a)
        return (f"answer {len(calls)}", "")

    _deliver(TURN, counted)
    with server._dedupe_lock:
        assert server._dedupe == {}, "a delivered turn was kept for replay"
    assert _deliver(TURN, counted) == (200, {"text": "answer 2", "thinking": ""})
    assert len(calls) == 2
