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

WORKDIR /app
# No requirements.txt -- the bridge itself is stdlib-only at runtime.
COPY bridge/ bridge/
COPY run.py .

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin bridge
USER bridge

CMD ["python", "run.py"]
