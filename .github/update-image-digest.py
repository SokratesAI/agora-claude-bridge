#!/usr/bin/env python3
"""Point this service's own container images at a new digest, and nothing else.

This replaces the blind `sed -i "s|image: .*|image: NEW|" manifest.yaml` that
committed the digest to every `-config` repo from 2026-07 until Cycle 852.
`sed` matched a *line*, so it rewrote every `image:` line in the file to this
repo's image whatever that line actually named. Nothing is broken today --
measured across all five config repos that have a manifest, every `image:`
line already names the repo's own image, and `agora-persona-runner-config`
holds two of them (the runner and nova-site) which genuinely do share one
digest. What the sed could not survive is the first sidecar, initContainer or
second workload: it would have silently retagged them to this service's image
and ArgoCD would have deployed that, with every check green.

The owner asked for a real YAML tool rather than a sharper regex (idea #230),
and not Kustomize -- too many files. So this parses the manifest as YAML to
decide *which* scalars to touch, and then rewrites those lines in place, which
is why `yq -i` is not used either: these manifests carry hundreds of lines of
hand-written comments and a whole-file re-emit would reformat the lot. Parsing
decides, the byte-level rewrite acts, and every other byte of the file is left
exactly as it was.

Two refusals, both deliberate, because this step's failure mode is the one
that deploys the wrong thing:

  * it exits non-zero if it matched nothing, since "no image line named this
    repo's image" means the manifest, the registry name or this script is
    wrong, and committing nothing quietly leaves the old digest running;
  * it exits non-zero rather than guessing on anything it cannot read.

A failed run is safe by construction: the digest never reaches the `-config`
repo, so ArgoCD keeps deploying the last image that did.

    python3 .github/update-image-digest.py <manifest> <registry/image@sha256:...>
"""

import sys

import yaml


def repository_of(reference):
    """The repository half of an image reference, without tag or digest.

    `ghcr.io/sokratesai/nova@sha256:ab` and `ghcr.io/sokratesai/nova:v2` both
    answer `ghcr.io/sokratesai/nova`. A colon inside the registry host's port
    (`localhost:5000/nova`) is not a tag, which is why the tag is only cut
    when the colon is after the last slash.
    """
    reference = reference.split("@", 1)[0]
    slash = reference.rfind("/")
    colon = reference.rfind(":")
    if colon > slash:
        reference = reference[:colon]
    return reference


def image_scalars(root):
    """Every `image:` value node in a composed document, in file order."""
    found = []
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if (
                    isinstance(key, yaml.ScalarNode)
                    and key.value == "image"
                    and isinstance(value, yaml.ScalarNode)
                ):
                    found.append(value)
                stack.append(value)
        elif isinstance(node, yaml.SequenceNode):
            stack.extend(node.value)
    return found


def rewrite(text, reference):
    """`text` with every image naming `reference`'s repository set to it.

    Returns (new_text, [(line_number, old_value)]) with line numbers 1-based
    for the report. Raises ValueError if the manifest cannot be parsed, if a
    matching value spans more than one line, or if nothing matched.
    """
    target = repository_of(reference)
    try:
        documents = list(yaml.compose_all(text))
    except yaml.YAMLError as exc:
        raise ValueError("could not parse the manifest as YAML: %s" % (exc,))

    matches = []
    for document in documents:
        if document is None:
            continue
        for node in image_scalars(document):
            if repository_of(node.value) != target:
                continue
            if node.start_mark.line != node.end_mark.line:
                raise ValueError(
                    "the image on line %d is written across several lines, "
                    "which this script will not rewrite"
                    % (node.start_mark.line + 1,)
                )
            matches.append(node)

    if not matches:
        raise ValueError(
            "no image in the manifest names %s, so there is nothing to "
            "update -- refusing rather than committing an unchanged file"
            % (target,)
        )

    lines = text.splitlines(keepends=True)
    changed = []
    # Descending, so an earlier rewrite cannot move a later line's marks.
    for node in sorted(matches, key=lambda n: n.start_mark.line, reverse=True):
        index = node.start_mark.line
        line = lines[index]
        start, end = node.start_mark.column, node.end_mark.column
        lines[index] = line[:start] + reference + line[end:]
        changed.append((index + 1, node.value))
    return "".join(lines), list(reversed(changed))


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(
            "usage: update-image-digest.py <manifest> <image reference>\n")
        return 2
    path, reference = argv[1], argv[2]
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    try:
        updated, changed = rewrite(text, reference)
    except ValueError as exc:
        sys.stderr.write("%s: %s\n" % (path, exc))
        return 1
    for line_number, old in changed:
        print("%s:%d %s -> %s" % (path, line_number, old, reference))
    if updated != text:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(updated)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
