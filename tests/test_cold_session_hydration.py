"""A conversation the bridge has never run a turn in has no session to
resume, so its first turn here is blind to everything already said in it.
Filed 2026-09-12 off a live case: `needs_input` opened a thread, wrote the
question through Agora's notify API, and the owner replied eleven hours
later -- the session that woke up to answer him saw one message out of
seventeen and had never seen its own question."""

from unittest.mock import patch

from bridge import server


def _run(history, session_id=None, **kwargs):
    captured = {}

    def fake_run_turn(**kw):
        captured.update(kw)
        return "ok", "", "sess-new"

    with patch.object(server, "get_session_id", return_value=session_id), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "his reply", history=history, **kwargs)
    return captured


PRIOR = [
    {"role": "assistant", "content": "Should I close idea #106?"},
    {"role": "user", "content": "What does it cost to keep it open?"},
    {"role": "assistant", "content": "About an hour a cycle."},
]


def test_a_cold_session_is_handed_the_conversation_so_far():
    message = _run(PRIOR)["message"]
    assert "Should I close idea #106?" in message
    assert "What does it cost to keep it open?" in message
    # The new message is still the new message, and it comes last.
    assert message.rstrip().endswith("his reply")


def test_a_resumed_session_is_handed_none_of_it():
    """The CLI already holds every one of those turns. Sending them again
    would be the same words twice, once as transcript and once as memory."""
    message = _run(PRIOR, session_id="sess-held")["message"]
    assert message == "his reply"
    assert "idea #106" not in message


def test_the_session_not_found_retry_hydrates_too():
    """The stored id pointing at a session the CLI no longer has is the path
    that actually loses a long-running chat -- the retry is every bit as
    blind as a first turn, and a fix that only covered `get_session_id`
    returning None would miss it."""
    seen = []

    def fake_run_turn(**kwargs):
        seen.append(kwargs.get("message"))
        if len(seen) == 1:
            raise server.ClaudeCliError(server.SESSION_NOT_FOUND)
        return "ok", "", "sess-2"

    with patch.object(server, "get_session_id", return_value="gone"), \
         patch.object(server, "set_session_id"), \
         patch.object(server, "clear_session_id"), \
         patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "his reply", history=PRIOR)
    assert seen[0] == "his reply"          # first attempt resumed, so no transcript
    assert "idea #106" in seen[1]          # the retry is cold and gets one


def test_stateless_is_never_hydrated():
    """`stateless` means "carry only what this prompt gives you" -- it is
    what Ask and the Evolve steps are for, and hydrating it would undo the
    thing the flag was built to do."""
    captured = {}

    def fake_run_turn(**kw):
        captured.update(kw)
        return "ok", "", "sess-1"

    with patch.object(server, "run_turn", side_effect=fake_run_turn):
        server.generate("conv-1", "system", "his reply", stateless=True, history=PRIOR)
    assert captured["message"] == "his reply"


def test_no_history_leaves_the_prompt_exactly_as_it_was():
    """Every caller that has never heard of the field, and the genuine
    first turn of a brand-new conversation, land here."""
    assert _run(None)["message"] == "his reply"
    assert _run([])["message"] == "his reply"


def test_an_empty_message_is_not_rendered_as_a_speaker():
    """A message whose text is blank (an image with no caption, a forgotten
    line the caller left in) must not print a bare "Owner:" with nothing
    after it -- that reads as the owner having said nothing on purpose."""
    rendered = server.render_prior_turns([
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "a real line"},
    ])
    assert "Owner:" not in rendered
    assert "You: a real line" in rendered


def test_nothing_sayable_renders_to_nothing():
    assert server.render_prior_turns([{"role": "user", "content": ""}]) == ""
    assert server.render_prior_turns([]) == ""
    assert server.render_prior_turns(None) == ""


def test_the_oldest_turns_are_dropped_to_fit_and_it_says_so():
    """The budget is on characters, not on turns: the caller's window is
    already bounded, and the danger here is an argv-sized prompt from a
    conversation that has run for months. It trims from the oldest end so
    the messages nearest the question survive."""
    big = [{"role": "user", "content": "x" * 30000},
           {"role": "assistant", "content": "y" * 30000},
           {"role": "user", "content": "the newest thing said"}]
    rendered = server.render_prior_turns(big)
    assert len(rendered) <= server.PRIOR_TURNS_BUDGET + 500
    assert "the newest thing said" in rendered
    assert "left out to fit" in rendered
    assert "x" * 30000 not in rendered


def test_a_transcript_that_fits_says_nothing_about_dropping():
    assert "left out to fit" not in server.render_prior_turns(PRIOR)


def test_roles_are_labelled_from_the_persona_point_of_view():
    rendered = server.render_prior_turns(PRIOR)
    assert "You: Should I close idea #106?" in rendered
    assert "Owner: What does it cost to keep it open?" in rendered


# --- the payload -> generate() wiring, which a mutation walked straight past --

def test_do_post_passes_history_through_to_generate():
    """Found by mutation: replacing the payload read with a literal `[]` broke
    nothing in the whole suite, so the route that carries the transcript was
    pinned at neither end. Every test above calls generate() directly."""
    from tests.test_bridge import _make_handler

    handler, sent = _make_handler({
        "conversation_id": "c1", "prompt": "hi",
        "history": [{"role": "user", "content": "said earlier"}],
    })
    captured = {}

    def fake_generate(conversation_id, system, prompt, **kwargs):
        captured["history"] = kwargs.get("history")
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    assert sent["status"] == 200
    assert captured["history"] == [{"role": "user", "content": "said earlier"}]


def test_do_post_history_is_empty_when_a_caller_omits_it():
    """A runner that predates this field gets exactly what it had before."""
    from tests.test_bridge import _make_handler

    handler, sent = _make_handler({"conversation_id": "c1", "prompt": "hi"})
    captured = {}

    def fake_generate(conversation_id, system, prompt, **kwargs):
        captured["history"] = kwargs.get("history")
        return "answer", ""

    with patch.object(server, "BRIDGE_TOKEN", ""), \
         patch.object(server, "generate", fake_generate):
        handler.do_POST()
    assert sent["status"] == 200
    assert captured["history"] == []
