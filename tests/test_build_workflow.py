"""The build workflow must serialise pipelines per branch.

`update-manifest` in build.yaml is a blind `sed` of whichever image digest
its own run produced, so two pipelines running in parallel deploy whichever
one *finished last*, which is not the same as whichever commit is newest.

Measured 2026-08-12: #41 and #40 merged four seconds apart, #40 was the
newer commit, and #41's build pushed its digest one second later and
overwrote it -- pinning agora-claude-bridge-config to the commit before the
/health/database endpoint #40 had just added. Both CI runs were green, the
image built, ArgoCD synced, and the deployed pod served a 404 for a feature
that was merged in main. Nothing anywhere reported an error.

Scope, stated precisely because the first draft of this docstring overclaimed
and a reviewer caught it: build.yaml is seeded by the Crossplane Composition
(githubservice-composition.yaml) with managementPolicies [Observe, Create,
LateInitialize], so Crossplane never overwrites this repo's hand-edited copy.
These tests guard *this repo only*. They are not seeded by the template --
the template ships a Node/vitest starter and its build.yaml runs `npm test`,
so a freshly created service would never run this file. The template got the
same concurrency block committed to it separately; nothing automatically
checks the two against each other.
"""

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "build.yaml"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def test_build_workflow_serialises_pipelines_per_branch():
    concurrency = _workflow().get("concurrency")
    assert concurrency, (
        "build.yaml declares no concurrency group -- two pushes to main run "
        "update-manifest in parallel and the last to finish wins, not the "
        "newest commit"
    )
    assert "github.ref" in concurrency["group"], (
        f"concurrency group {concurrency['group']!r} must be per-branch, or "
        "every pull_request build queues behind main"
    )


def test_in_flight_builds_are_never_cancelled():
    # Killing a run partway through update-manifest is the failure the
    # concurrency group exists to prevent, not an optimisation to add later.
    assert _workflow()["concurrency"]["cancel-in-progress"] is False
