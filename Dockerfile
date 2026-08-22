FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node -- required to install/run the `claude` CLI (it's a Node package).
# Pinned major version, not "latest", same reasoning as agora-persona-runner's
# pinned kubectl/gh versions -- a rebuild months from now shouldn't silently
# pick up a different major version.
ARG NODE_MAJOR=20
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
ARG CLAUDE_CODE_VERSION=2.1.226
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
ARG KUBECTL_VERSION=v1.36.2
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
