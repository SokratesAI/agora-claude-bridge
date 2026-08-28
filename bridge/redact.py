"""Strips live credentials out of anything narrated back to the caller.

Cycle 20 read the CLI's own credentials file to see what auth we had and
printed it. The full OAuth access *and* refresh tokens went into that
session's tool output. They did not reach the owner's conversation only
because the output half of the narration was still sitting in an unmerged
PR -- the one this module ships with. Merging that PR without this one
would turn a careless `print` into a published, permanently stored secret.

This is the narrow case where a filter earns its place: the danger is
nameable, it has already happened once, and the material is verbatim
machine output that nobody reviewed before it was sent. It is deliberately
NOT a general "looks sensitive" heuristic -- every pattern below matches a
credential format, not a topic, and a redaction is always visible as
`[redacted: <what>]` rather than a silent deletion. The owner's standing rule
is that nothing gets thrown away to make the UI tidier; the answer to "too
much output" is an interface, not a filter. A live token is the exception,
and it is the only one.

Applied at the single point everything leaves for the runner
(activity.py's sender), so a new call site cannot bypass it by forgetting.
"""

import os
import re

# (label, pattern). The label is what the reader sees in place of the
# secret, so it says what was removed without saying what it was.
_PATTERNS = (
    # Anthropic API keys and, the ones that actually leaked, the OAuth
    # access/refresh tokens the CLI stores (sk-ant-oat01-/sk-ant-ort01-).
    ("anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    # GitHub: PATs (classic + fine-grained), OAuth, user, server, refresh.
    ("github token", re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}")),
    ("github token", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    # A JWT is three base64url segments; the middle one starts a JSON
    # object, so a real one begins eyJ. Session cookies and k8s
    # ServiceAccount tokens both take this shape.
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("aws key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    # Google API keys, in both of the two shapes this estate has seen: the
    # classic `AIza` + 35 base64url characters (39 long), and the newer
    # `AQ.` + 50 that Google AI Studio issues now. This is the one
    # credential here that has been seen printed with no name beside it --
    # `GEMINI_API_KEY` in the runner pod carries a trailing newline,
    # `urllib` refuses a header value containing one, and the exception it
    # raises quotes the whole header value back. A traceback has no `NAME=`
    # in it, so the value pattern at the bottom of this table cannot see it,
    # and every other pattern here is for a different vendor.
    #
    # The second alternative is the reason this comment is long. Cycle 503
    # wrote the `AIza` half first, from the documented format and gitleaks'
    # own `gcp-api-key` rule, then measured the live key in the runner pod
    # rather than trusting either: it is 53 characters, starts `AQ.`, and
    # the `AIza` pattern does not match it. A rule that covers the format
    # in the documentation and not the key on the box is a filter that
    # reports itself working while guarding nothing. Measured without
    # printing the value -- length, prefix and character class only.
    ("google api key", re.compile(
        r"AIza[0-9A-Za-z_\-]{35}"
        r"|AQ\.[0-9A-Za-z_\-]{40,}")),
    ("private key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    )),
    # `printenv`, a .env file, a k8s secret dumped as YAML/JSON: the value
    # is unguessable but the NAME beside it is not. Only the value is
    # replaced -- the name stays, because knowing that ANTHROPIC_API_KEY is
    # set is exactly the kind of thing he wants to be able to see.
    #
    # Two of those three shapes were missed until Cycle 170, and the drift
    # probes in tools/sync_contract.py found both on their first run:
    #
    #   * JSON quotes the NAME too, so `"couchdb_password": "x"` put a `"`
    #     between the name and the colon and nothing matched. The optional
    #     quote stays inside group 2, so the replacement puts it back and the
    #     document is still parseable.
    #   * `_PASS` is not `PASSWD` or `PASSWORD`, and `CDB_PASS` is the name
    #     this very system holds its CouchDB password under. It is spelled
    #     `_PASS` rather than `PASS` on purpose: a bare `pass:` is an
    #     ordinary English word, and "second pass: completed" is exactly the
    #     over-redaction the owner's keep-everything rule forbids.
    ("value", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|_PASS|API[_-]?KEY|ACCESS[_-]?KEY|CREDENTIAL)S?)"
        r"(\"?\s*[=:]\s*\"?)"
        # The lookahead is load-bearing and was not in the pre-Cycle-170
        # version, because that version could not reach this position at all
        # in JSON. `_PATTERNS` runs in order, so by the time this one sees
        # the CLI credentials file the anthropic-key pattern has already
        # replaced the token with `[redacted: anthropic key]` -- and
        # `[redacted:` is 10 characters none of which are excluded below, so
        # this pattern matched the marker as if it were the value and
        # produced `[redacted: value] anthropic key]`. Caught by
        # test_tool_output_is_redacted_on_the_way_out in the bridge, which
        # had been green for weeks and went red on the widening.
        r"((?!\[redacted:)[^\s\"',}]{8,})"
    )),
)


# The names above cover a credential that has a *format*. Three of the ones
# actually mounted here do not, and Cycle 560 measured that on the live pods
# rather than reasoning about it: `redact()` returned the bare value of
# AGORA_TOKEN (64 chars), COUCHDB_PASSWORD/CDB_PASS (16) and TINYFISH_API_KEY
# (44, `sk-` but not `sk-ant-`) completely unaltered. Those are the token that
# can act as any persona on Agora, the password to the CouchDB holding both
# Nova's database and the owner's Obsidian vault, and a third-party API key.
# No shape rule can ever catch them, because a random 16-character password
# has no shape -- that is the whole point of it.
#
# So this pass matches by *value* instead. The process publishing the text is
# the process holding the secret, so it can look the literal up rather than
# guess at it, and a literal match cannot be defeated by a format nobody
# documented. This is the same move as the sandbox's `credentials.mode: "mask"`
# that idea #106 points at, done at the one boundary this loop actually
# controls today: the CLI's own masking needs the sandbox proxy turned on to
# substitute on egress, and turning that on would confine this loop's shell.
#
# Selected by name, using the same vocabulary as the `value` pattern above so
# there is one list of what "a secret" is called and not two.
_SECRET_NAME = re.compile(
    r"(?i)^[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|_PASS|API[_-]?KEY"
    r"|ACCESS[_-]?KEY|CREDENTIAL)S?$"
)

# A floor, and it is measured rather than chosen for comfort. The shortest
# secret-named value on either pod is 16 characters (COUCHDB_PASSWORD); the
# danger below this line is real and in the other direction -- an env var
# named like a secret holding `true` or `none` would blank that word out of
# every sentence this loop publishes, which is exactly the over-redaction the
# owner's keep-everything rule forbids. 8 sits an octave clear of both.
_MIN_SECRET_LEN = 8


def _secret_literals(environ=None):
    """`(name, value)` for every env var whose name says it holds a secret.

    Longest value first, so a secret that contains another one is replaced
    whole rather than leaving the tail of the shorter match behind.

    The value is stripped, and that is the whole handling of surrounding
    whitespace rather than an oversight. GEMINI_API_KEY on the runner pod
    carries a trailing newline and it is the *unstripped* value `urllib` quotes
    back in the exception this idea was filed over -- but the stripped form is a
    substring of the raw one, so matching on it covers both. A first version
    carried the raw value as well; the mutation that removed it survived every
    test, which is the tell that it was a branch doing no work.
    """
    env = os.environ if environ is None else environ
    out = []
    for name, value in env.items():
        if not isinstance(value, str) or not _SECRET_NAME.match(name or ""):
            continue
        literal = value.strip()
        if len(literal) >= _MIN_SECRET_LEN and (name, literal) not in out:
            out.append((name, literal))
    out.sort(key=lambda pair: len(pair[1]), reverse=True)
    return out


def redact(text, environ=None):
    """`text` with any credential-shaped run replaced by a visible marker.

    Returns non-strings unchanged so callers don't have to type-check
    before handing over whatever a tool returned. `environ` is for tests --
    production reads the real environment, so nothing has to be configured
    and a credential added to a pod is covered the moment it is mounted.
    """
    if not isinstance(text, str) or not text:
        return text
    for name, literal in _secret_literals(environ):
        if literal in text:
            text = text.replace(literal, "[redacted: %s]" % (name,))
    for label, pattern in _PATTERNS:
        if label == "value":
            text = pattern.sub(rf"\1\2[redacted: {label}]", text)
        else:
            text = pattern.sub(f"[redacted: {label}]", text)
    return text
