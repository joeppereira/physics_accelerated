#!/bin/bash
# 128G SerDes Design Cycle: Physics-to-Cognitive (Local Orchestration)

# Ensure output directory exists
mkdir -p reports plots

echo "🚀 [1/4] Starting Physics Core: Generating 3nm Ground Truth..."
# Using local dummy_gen as the Physics Core proxy for data generation
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/python3 src/dummy_gen.py

echo "🧠 [2/4] Starting Cognitive Optimizer: Training FNO Surrogate..."
./venv/bin/python3 src/train.py --epochs 50

echo "🧬 [3/4] Running GEPA Evolution: Finding Optimal Config..."
./venv/bin/python3 src/gepa.py --target_loss -36.0 > reports/gepa_results.txt

echo "📊 [4/4] Generating Architectural Yield Reports..."
./venv/bin/python3 src/stats_plotter.py

echo "✅ Full Cycle Complete. See reports/gepa_results.txt for the Golden Config."
