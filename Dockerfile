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

RUN npm install -g @anthropic-ai/claude-code

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

WORKDIR /app
# No requirements.txt otherwise -- the bridge itself is stdlib-only at runtime.
COPY bridge/ bridge/
COPY run.py .

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin bridge
USER bridge

CMD ["python", "run.py"]
