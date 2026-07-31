"""conversation_id -> Claude Code session_id mapping, persisted to a JSON
file on the same PVC as CLAUDE_HOME so it survives pod restarts.

One file, one lock -- this service is deliberately single-instance (see
cli.py's module docstring for why), so a plain in-process lock is enough;
no database, no distributed locking needed.
"""
import json
import os
import threading

from bridge.config import SESSIONS_FILE
from bridge.log import log

_lock = threading.Lock()


def _load():
    if not os.path.exists(SESSIONS_FILE):
        return {}
    try:
        with open(SESSIONS_FILE) as f:
            return json.load(f)
    except Exception as e:
        log(f"sessions: failed to read {SESSIONS_FILE}, starting empty: {e}")
        return {}


def _save(data):
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    tmp = SESSIONS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, SESSIONS_FILE)  # atomic on POSIX -- no torn reads


def get_session_id(conversation_id):
    with _lock:
        return _load().get(conversation_id)


def set_session_id(conversation_id, session_id):
    with _lock:
        data = _load()
        data[conversation_id] = session_id
        _save(data)


def clear_session_id(conversation_id):
    """Used when the CLI reports SESSION_NOT_FOUND -- the on-disk session
    file is gone (e.g. CLAUDE_HOME was ever recreated) even though our own
    mapping still points at it; drop the stale mapping so the next turn
    starts a fresh session instead of failing forever."""
    with _lock:
        data = _load()
        if data.pop(conversation_id, None) is not None:
            _save(data)
