#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_route_c_core.sh"
"${SCRIPT_DIR}/run_route_c_ablation.sh"
