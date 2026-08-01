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

WORKDIR /app
# No requirements.txt -- the bridge itself is stdlib-only at runtime.
COPY bridge/ bridge/
COPY run.py .

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin bridge
USER bridge

CMD ["python", "run.py"]
