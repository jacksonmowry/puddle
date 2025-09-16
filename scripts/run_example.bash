#!/usr/bin/env bash

set -euo pipefail

bin/generate_reservoir -p 0.15 -f 20 -c 4 | framework-open/bin/network_tool >reservoirs/reservoir.json
bin/classify reservoirs/reservoir.json datasets/quadrant/data.csv datasets/quadrant/labels.csv
