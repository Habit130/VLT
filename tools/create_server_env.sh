#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.server.linux.cuda118.yml}"
TMP_CONDARC="$(mktemp)"
trap 'rm -f "${TMP_CONDARC}"' EXIT

cat > "${TMP_CONDARC}" <<'EOF'
channels:
  - conda-forge
default_channels: []
custom_channels: {}
show_channel_urls: true
channel_priority: flexible
EOF

CONDA_NO_PLUGINS=true \
CONDARC="${TMP_CONDARC}" \
conda env create \
  --solver classic \
  --file "${ENV_FILE}"
