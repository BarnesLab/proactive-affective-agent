#!/bin/bash
# Phase 0 pipeline: build participant roster then compute home locations.
# Run from any directory; this script anchors to the project root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================"
echo "Phase 0 — Sensing Data Infrastructure"
echo "Project root: $PROJECT_ROOT"
echo "============================================================"

cd "$PROJECT_ROOT"

echo ""
echo "[1/2] Building participant roster..."
python scripts/offline/build_participant_roster.py

echo ""
echo "[2/2] Computing home locations..."
python scripts/offline/compute_home_locations.py

echo ""
echo "============================================================"
echo "Phase 0 complete."
echo "Outputs written to: data/processed/hourly/"
echo "============================================================"
