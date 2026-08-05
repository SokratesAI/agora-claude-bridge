"""Strips live credentials out of anything narrated back to the caller.

Cycle 20 read the CLI's own credentials file to see what auth we had and
printed it. The full OAuth access *and* refresh tokens went into that
session's tool output. They did not reach Edvard's conversation only
because the output half of the narration was still sitting in an unmerged
PR -- the one this module ships with. Merging that PR without this one
would turn a careless `print` into a published, permanently stored secret.

This is the narrow case where a filter earns its place: the danger is
nameable, it has already happened once, and the material is verbatim
machine output that nobody reviewed before it was sent. It is deliberately
NOT a general "looks sensitive" heuristic -- every pattern below matches a
credential format, not a topic, and a redaction is always visible as
`[redacted: <what>]` rather than a silent deletion. Edvard's standing rule
is that nothing gets thrown away to make the UI tidier; the answer to "too
much output" is an interface, not a filter. A live token is the exception,
and it is the only one.

Applied at the single point everything leaves for the runner
(activity.py's sender), so a new call site cannot bypass it by forgetting.
"""
import re

# (label, pattern). The label is what the reader sees in place of the
# secret, so it says what was removed without saying what it was.
_PATTERNS = (
    # Anthropic API keys and, the one that actually leaked, the OAuth
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
    ("private key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    )),
    # `printenv`, a .env file, a k8s secret dumped as YAML/JSON: the value
    # is unguessable but the NAME beside it is not. Only the value is
    # replaced -- the name stays, because knowing that ANTHROPIC_API_KEY is
    # set is exactly the kind of thing he wants to be able to see.
    ("value", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API[_-]?KEY|ACCESS[_-]?KEY|CREDENTIAL)S?)"
        r"(\s*[=:]\s*\"?)"
        r"([^\s\"',}]{8,})"
    )),
)


def redact(text):
    """`text` with any credential-shaped run replaced by a visible marker.

    Returns non-strings unchanged so callers don't have to type-check
    before handing over whatever a tool returned.
    """
    if not isinstance(text, str) or not text:
        return text
    for label, pattern in _PATTERNS:
        if label == "value":
            text = pattern.sub(rf"\1\2[redacted: {label}]", text)
        else:
            text = pattern.sub(f"[redacted: {label}]", text)
    return text
