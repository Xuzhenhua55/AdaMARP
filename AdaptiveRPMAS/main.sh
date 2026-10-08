#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python}" "$BASE_DIR/main.py" --config-file "$BASE_DIR/adaptiveRPMAS-config-en.json" "$@"
