#!/usr/bin/env bash
# Start the image this pipeline is about to deploy, and prove it can import
# what it ships.
#
# Nothing in either build pipeline ran the image it built until 2026-09-03.
# On 2026-09-02 agora-persona-runner shipped an image that crashed on
# `import yaml` -- a PR added the import, the Dockerfile had no pip install
# step, and the unit suite was green throughout because it runs on a GitHub
# runner that installs pyyaml for its own reasons. That deployment is
# single-replica with strategy Recreate and no fallback, and Nova's heartbeat
# runs in it, so the one thing that watches this system was the thing that was
# down. Ten hours, found by hand.
#
# This file is deliberately byte-identical in agora-persona-runner and
# agora-claude-bridge, which is why it discovers the package to sweep rather
# than naming one: `tools/sync_contract.py` compares the two build-push jobs
# as parsed YAML and refuses a difference, and a step that hardcoded
# `agora_runner` would be a difference. Both images are the same shape -- a
# python base, one or more packages copied into /app, `python run.py` as the
# command -- so "import every module of every package in the workdir" is one
# check that is honest about both.
#
# It imports rather than starting run.py because the entrypoint needs
# credentials and a reachable Agora, while an import sweep needs neither:
# measured 2026-09-03, all 71 modules of agora_runner import with an empty
# environment and no network. So this runs with --network none, which also
# means a smoke test can never accidentally talk to production.
set -euo pipefail

IMAGE="${1:?usage: smoke-test.sh <image ref>}"

docker pull -q "$IMAGE"
docker run --rm --network none --entrypoint python "$IMAGE" -c '
import importlib, os, pkgutil, sys

packages = sorted(
    entry for entry in os.listdir(".")
    if os.path.isfile(os.path.join(entry, "__init__.py"))
)
if not packages:
    sys.exit("no python package found in %s -- this smoke test is not looking at the code" % os.getcwd())

broken = []
found = 0
for name in packages:
    found += 1
    try:
        package = importlib.import_module(name)
    except Exception as exc:
        # The package __init__ itself is the likeliest thing to fail, because
        # it is what re-exports the rest -- report it like any other module
        # rather than dying with a traceback the log has to be read to parse.
        broken.append("  %s: %s: %s" % (name, type(exc).__name__, exc))
        continue
    for module in pkgutil.walk_packages(package.__path__, name + "."):
        found += 1
        try:
            importlib.import_module(module.name)
        except Exception as exc:
            broken.append("  %s: %s: %s" % (module.name, type(exc).__name__, exc))

if broken:
    sys.exit("the image cannot import what it ships:\n" + "\n".join(broken))
if found < 10:
    sys.exit("only %d module(s) found in %s -- the sweep is not looking at the code" % (found, packages))
print("smoke test: %d module(s) in %s imported inside the image" % (found, ", ".join(packages)))
'
