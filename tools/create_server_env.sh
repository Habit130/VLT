#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${1:-environment.server.linux.cuda118.yml}"

CONDA_NO_PLUGINS=true \
conda env create \
  --solver classic \
  --override-channels \
  -c conda-forge \
  -c nodefaults \
  -f "${ENV_FILE}"
