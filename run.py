"""Entrypoint -- runs the bridge server in the foreground (no separate
background-thread indirection needed like agora-persona-runner's poll loop,
this service has nothing else to do)."""
from bridge.server import start_server

if __name__ == "__main__":
    start_server()
