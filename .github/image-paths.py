#!/usr/bin/env python3
"""Can this push possibly have changed the image this repo builds?

`build-push` runs on every merge to main and takes about four and a half
minutes, and every run of it commits a digest to the paired -config repo and
rolls the Deployments that repo describes. Measured Cycle 844: 79 of the last
150 merges to main changed no file this image ships, and all 150 rolled
anyway.

Cycle 844's fix was a BuildKit layer cache, on the reasoning that an image
whose contents did not change would keep its digest and the pipeline's
existing "no manifest change, skipping commit" branch would then be taken.
**The cache works and the reasoning was still wrong.** Measured Cycle 853 on
run 33781833369 -- merge #685, which touched only `tools/` and `tests/`:
BuildKit reported `CACHED` on all nine of our layers, and the pushed manifest
still carried a brand-new digest, and -config still took a commit 4 minutes
later. The cause is one step above the layers: `docker/metadata-action`
stamps `org.opencontainers.image.created` and `.revision` into the image
config, so that blob holds this build's wall-clock time and this commit's
SHA. The manifest digest covers the config blob. A new commit therefore
*always* produces a new digest, cache or no cache, and the skip branch can
never be reached while those labels exist. Dropping the labels would buy the
skip at the price of not knowing which commit any deployed image came from,
which is a bad trade -- so the decision has to be made before the build, not
after it, which is what this file is.

The list of paths is read out of the Dockerfile rather than written down
here, because a written-down list is a second copy of the truth and goes
stale exactly the way the pinned versions this repo watches do. The failure
that matters is the too-narrow one -- a real change that silently never
deploys -- so every branch that cannot answer confidently answers "build":
an unparseable COPY, an empty diff, a Dockerfile that cannot be read.

Deliberately included beyond the COPY sources, because neither is derivable:
the Dockerfile itself (its RUN lines install pinned tools into the image) and
`.github/workflows/build.yaml` (a change to how the build is invoked can
change what it produces). `.dockerignore` is included when present for the
same reason -- it decides what a COPY actually copies.
"""

import argparse
import fnmatch
import os
import sys


class UnreadableCopy(Exception):
    """A COPY/ADD line this parser does not understand.

    Raised rather than skipped: a COPY we cannot read is a set of paths we
    cannot watch, and silently ignoring it is the too-narrow filter.
    """


# Not derivable from the Dockerfile; see the module docstring.
ALWAYS = ("Dockerfile", ".dockerignore", ".github/workflows/build.yaml")


def _logical_lines(text):
    """Dockerfile lines with `\\` continuations joined and comments dropped."""
    out, buf = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not buf and (not line or line.startswith("#")):
            continue
        if line.endswith("\\"):
            buf += line[:-1].strip() + " "
            continue
        out.append((buf + line).strip())
        buf = ""
    if buf:
        out.append(buf.strip())
    return out


def copy_sources(dockerfile_text):
    """Every build-context path a COPY or ADD instruction reads.

    A `COPY --from=...` is skipped: its source is another build stage, not a
    file in this repo.
    """
    sources = []
    for line in _logical_lines(dockerfile_text):
        head, _, rest = line.partition(" ")
        if head.upper() not in ("COPY", "ADD"):
            continue
        if rest.lstrip().startswith("["):
            # JSON-array form. Parsing it is easy; the point is that nothing
            # in this repo uses it, so an untested branch would be worse than
            # an honest refusal.
            raise UnreadableCopy(line)
        tokens = rest.split()
        flags = [t for t in tokens if t.startswith("--")]
        operands = [t for t in tokens if not t.startswith("--")]
        if any(f.startswith("--from=") for f in flags):
            continue
        if len(operands) < 2:
            raise UnreadableCopy(line)
        sources.extend(operands[:-1])
    return sources


def image_paths(dockerfile_text):
    """Sorted repo paths a change to which can change the built image."""
    return sorted(set(copy_sources(dockerfile_text)) | set(ALWAYS))


def _matches(pattern, changed):
    """Does `changed` (a repo-relative file path) fall under `pattern`?"""
    pattern = pattern.rstrip("/")
    if pattern in (".", ""):
        return True
    if fnmatch.fnmatch(changed, pattern):
        return True
    return changed.startswith(pattern + "/")


def affects_image(paths, changed_files):
    """True if any changed file falls under one of `paths`.

    An empty `changed_files` is True, not False: "the diff told me nothing"
    and "the diff told me nothing changed" arrive here identically, and only
    one of them is safe to act on.
    """
    changed = [c.strip() for c in changed_files if c.strip()]
    if not changed:
        return True
    return any(_matches(p, c) for p in paths for c in changed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dockerfile", default="Dockerfile")
    parser.add_argument(
        "--changed-files",
        help="file holding one changed repo path per line, or - for stdin",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the watched paths and exit",
    )
    args = parser.parse_args(argv)

    try:
        with open(args.dockerfile, encoding="utf-8") as fh:
            paths = image_paths(fh.read())
    except (OSError, UnreadableCopy) as exc:
        # Cannot tell -> build. Loud, because a repo that always builds looks
        # exactly like a repo with no filter at all.
        print("::warning::cannot read %s (%s); building" % (args.dockerfile, exc),
              file=sys.stderr)
        if args.list:
            return 1
        print("true")
        return 0

    if args.list:
        for path in paths:
            print(path)
        return 0

    if args.changed_files == "-":
        changed = sys.stdin.read().splitlines()
    elif args.changed_files:
        with open(args.changed_files, encoding="utf-8") as fh:
            changed = fh.read().splitlines()
    else:
        parser.error("one of --changed-files or --list is required")

    print("true" if affects_image(paths, changed) else "false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
