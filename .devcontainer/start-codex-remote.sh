#!/usr/bin/env bash

set -u

codex_bin="${HOME}/.local/bin/codex"

if [[ ! -x "${codex_bin}" ]]; then
  echo "Installing the official standalone Codex CLI..."
  if ! curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh; then
    echo "Codex installation failed; remote control was not started." >&2
    exit 0
  fi
fi

if ! "${codex_bin}" login status >/dev/null 2>&1; then
  echo "Codex is not signed in. Run: ${codex_bin} login --device-auth" >&2
  exit 0
fi

for attempt in 1 2 3; do
  if "${codex_bin}" remote-control start; then
    exit 0
  fi

  if [[ "${attempt}" -lt 3 ]]; then
    echo "Codex remote control is not ready yet; retrying..." >&2
    sleep 5
  fi
done

echo "Codex remote control could not be started after 3 attempts." >&2
exit 0
