#!/bin/bash
# graph_vlm_rag — one-shot demo
#
# Usage: bash scripts/run_demo.sh
#
# What it does:
# 1. Ingest the sample PDF
# 2. Run the eval questions against the indexed data

set -e

cd "$(dirname "$0")/.."

echo "================================================"
echo "  graph_vlm_rag Demo"
echo "================================================"
echo ""

# Step 1: Ingest the sample PDF
echo ">>> Step 1: Ingest sample PDF"
echo ""
python3 -m graph_vlm_rag ingest assets/sample.pdf

echo ""
echo ">>> Step 2: Run evaluation questions"
echo ""
python3 -m graph_vlm_rag eval

echo ""
echo "================================================"
echo "  Demo complete!"
echo "================================================"