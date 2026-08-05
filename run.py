"""Entrypoint -- installs the drain handlers, then runs the bridge server
until SIGTERM.

start_server keeps this thread parked rather than serving on it: it hands
serve_forever to a worker so the main thread stays free to notice SIGTERM
and run the shutdown (see bridge/server.py). start_server returns, rather
than never returning, once the drain is done."""
from bridge.credentials import bootstrap_credentials
from bridge.git_setup import bootstrap_git
from bridge.server import install_signal_handlers, start_server

if __name__ == "__main__":
    # Before the server starts, and on the main thread -- signal.signal()
    # only works there. Without it SIGTERM kills a turn in flight.
    install_signal_handlers()
    bootstrap_credentials()
    bootstrap_git()
    start_server()
