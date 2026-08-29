FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node -- required to install/run the `claude` CLI (it's a Node package).
# Pinned major version, not "latest", same reasoning as agora-persona-runner's
# pinned kubectl/gh versions -- a rebuild months from now shouldn't silently
# pick up a different major version.
#
# A pinned major goes end-of-life on a date, and nothing here reads that date:
# this said 20 until 2026-08-27, four months after Node 20 stopped receiving
# security patches on 2026-04-30. Dependabot does not read this line -- it
# reads manifests and lockfiles -- so the pin has to be checked against
# nodejs/Release's schedule by hand or by a check built for it (idea #151).
# 24 is the active LTS and is supported to April 2028.
ARG NODE_MAJOR=24
RUN curl -fsSL https://deb.nodesource.com/setup_${NODE_MAJOR}.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# The `claude` CLI itself -- pinned for the same reason NODE_MAJOR and
# KUBECTL_VERSION above are, which this line used to be the one exception to.
#
# Unpinned was not "tracks latest"; it was "frozen, invisibly". Nothing above
# this line changes between builds, so Docker's layer cache never re-ran the
# install, and the version stayed at whatever it was when the layer was first
# built while looking like it followed the registry. Measured 2026-08-10: the
# running pod had 2.1.197 and the registry had 2.1.226 -- 29 releases, among
# them two long-session performance fixes, a memory-growth fix for truncated
# MCP tool outputs, and `--forward-subagent-text`.
#
# Bumping this is a deliberate, revertible change, and it deserves to be: the
# loop that maintains this repo runs *inside* this binary, so a breaking change
# to the stream-json contract in bridge/cli.py takes out the cycle that would
# otherwise fix it. Before bumping, re-run the check that justified this one --
# run the same prompt through both versions with
# `--print --output-format stream-json --verbose` and diff the sequence of
# event `type`s and content-block `type`s. On 2.1.197 vs 2.1.226 they were
# identical, and every field cli.py reads (`id`/`name`/`input` on tool_use,
# `tool_use_id`/`is_error` on tool_result, `rate_limit_info`, `session_id`)
# was present under the same name.
#
# Re-run 2026-08-25 for 2.1.226 -> 2.1.245, same prompt through both binaries.
# The *set* of message and content-block types is identical; 2.1.245 emits more
# `system` events and orders `rate_limit_event` earlier, neither of which cli.py
# keys on. Content-block field names are byte-identical on all four block types.
# Top-level keys are a superset with one removal: `messaging_socket_path` is
# gone, and it is read nowhere in this repo (grep: 0 hits). Every field the
# comment above names is still present under the same name.
#
# Re-run 2026-08-29 for 2.1.245 -> 2.1.251, and this time through BOTH input
# paths rather than one. tools.changelog_watch read 215 entries across the
# five releases in that gap and marked exactly one: 2.1.251 fixes
# `--input-format stream-json` merging client-injected assistant tool calls
# that carry no message id. That is a flag cli.py passes.
#
# write_stream_json_input sends a single `user` event and no
# assistant blocks at all, so that specific bug was never reachable from here
# -- but it is our parser, so the check was run over `--print <text>` and over
# `--input-format stream-json` on stdin separately. Both: event/subtype/block
# sequence identical, top-level key sets identical (no removal this time), and
# `text`/`tool_use`/`tool_result` block key sets identical. Every field cli.py
# reads was present -- `id`/`name`/`input`, `tool_use_id`/`is_error`,
# `rate_limit_info`, `session_id`.
#
# That diff is only worth quoting because it was shown capable of failing:
# dropping `session_id` and `tool_use.input` out of the 2.1.251 capture made
# it report six differences. A clean diff from an instrument nobody mutated
# is a positive result guaranteed in advance.
#
# The `engines.node >= 22` mismatch this comment used to record is closed:
# NODE_MAJOR above is 24, so the CLI now runs inside its declared range
# rather than one major below it. Re-measured 2026-08-29 on a real v24.20.0
# binary -- `npm install --prefix <tmp> @anthropic-ai/claude-code@2.1.251`
# exits 0 with no engine warning and the installed `claude --version` prints
# `2.1.251 (Claude Code)`. A prefix install, not the `-g` on the line below:
# the running pod's global binary is what this cycle was speaking through.
ARG CLAUDE_CODE_VERSION=2.1.251
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}

# kubectl -- 2026-08-01 design call: this service should be as capable as
# an interactive Claude Code session, including real cluster access via
# terminal/Bash, not silently limited. Pinned version, same reasoning as
# agora-persona-runner's own pinned kubectl -- a rebuild months from now
# shouldn't silently pick up a different major version. RBAC is scoped at
# the API-server level (see agora-claude-bridge-config's ClusterRole,
# mirroring agora-persona-runner-read: cluster-wide, read-only, Secrets
# excluded from every rule) -- this binary alone grants nothing beyond
# whatever the pod's ServiceAccount is actually bound to.
ARG KUBECTL_VERSION=v1.35.8
RUN curl -fsSLo /usr/local/bin/kubectl \
    "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" \
    && chmod +x /usr/local/bin/kubectl

# git + gh -- 2026-08-01: the Evolve workflow's personas now run entirely
# through this bridge (see the vault's Agora/_context.md) and need the same
# real git clone/commit/push + gh pr create/diff/checks/merge workflow this
# very session uses, since Agora's purpose-built github_read/create_pr/
# merge_pr tools don't apply to a claude-cli persona at all. Credentials
# come from GH_TOKEN at runtime (bridge/git_setup.py), not baked in here.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN (type -p wget >/dev/null || apt-get update && apt-get install -y --no-install-recommends wget) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -nv -O /etc/apt/keyrings/githubcli-archive-keyring.gpg https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# openssh-client -- so `tools.nas_health` can see the NAS from THIS pod.
#
# The NAS is the one thing Edvard calls highest priority, and its apps are the
# part of it nothing checks unprompted. `tools.preflight` runs on this pod
# every cycle and its `nas_health` check can only judge whether the box is up:
# it opens a TCP connection to port 22 and reads the SSH banner. The half it
# cannot do is Sonarr and Radarr, because `allow-nas-ssh-egress` opens port 22
# and nothing else, so those apps are reachable only by running `curl` ON the
# NAS over an SSH hop -- and this image has no `ssh` binary at all. The runner
# pod has one, which is why `tools.nas status` answers there; what it does not
# have is anything scheduled, so nobody runs it.
#
# This is one of the two changes that close that. The other is mounting the
# existing `nas-ssh-key` secret (already sealed, already in the `agents`
# namespace for the runner) at /etc/nas-ssh on this deployment, in
# agora-claude-bridge-config. Neither is any use without the other, and
# nas_health prints `CANNOT SEE FROM THIS POD` rather than failing while
# either is missing.
#
# Its own layer near the bottom, same reasoning as tini below: the npm,
# kubectl and gh layers above stay cached for a one-package install. Do not
# fold it into one of those apt lines -- a future tidy-up of an unrelated
# package list is exactly how this would disappear again, and the symptom is
# a check going quiet rather than a build going red.
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*

# xxhash -- LiveSync's real chunk-id algorithm (bridge/vault_tool.py falls
# back to sha256 if this is ever missing, same as agora_runner/vault.py).
RUN pip install --no-cache-dir xxhash

# tini -- an init at PID 1, whose whole job is reaping orphans.
#
# Without it `python run.py` is PID 1. Any process whose parent dies is
# reparented to PID 1, and PID 1 is expected to wait() on it; this process only
# ever waits on pids it spawned itself, so every adopted orphan stayed a zombie
# forever -- and a zombie still holds a pid slot.
#
# Measured on the live pod 2026-08-22 at 27h uptime, which is what turns this
# from a known container footgun into this pod's actual cause of death:
# /sys/fs/cgroup/pids.current was 9248 while PID 1 itself held 33MB RSS and 5
# threads, and every sampled entry of /proc/1/task/1/children (2012, 60045,
# 117134 -- old, middle and recent) was `git` in state Z with PPid 1. The pod
# had hit its pids cgroup limit, so nothing in it could fork: the readiness
# probe had been failing for 57 minutes with EOF, the server logged
# `RuntimeError: can't start new thread` out of socketserver on every request,
# and every shell command died with `fork: Resource temporarily unavailable`.
# Memory and CPU were nowhere near their limits (576Mi/2Gi, 40m/1), which is
# why this reads as an idle, healthy pod on every dashboard.
#
# tini only reaps what reparents to *it*. The bridge's own subprocesses stay
# children of the python process and are still waited on by subprocess.Popen,
# so cli.run_turn's exit-code and timeout handling is untouched. An in-process
# SIGCHLD reaper would not be safe here for exactly that reason: it could reap
# the `claude` subprocess before proc.wait() does, and Popen reports a stolen
# child as exit 0 -- silently turning a failed turn into a successful one.
#
# Deliberately NOT `-g`. That forwards signals to the whole process group,
# which would kill the in-flight `claude` turn that bridge/server.py's drain
# exists to let finish. Plain `--` signals only the direct child, which is the
# behaviour the drain was written against.
#
# Its own layer, near the bottom, so the npm/kubectl/gh layers above stay
# cached rather than rebuilding for a one-package install.
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# No requirements.txt otherwise -- the bridge itself is stdlib-only at runtime.
COPY bridge/ bridge/
COPY run.py .

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin bridge
USER bridge

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "run.py"]
