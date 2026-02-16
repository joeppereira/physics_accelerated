#!/bin/bash
# 128G SerDes Design Cycle: Physics-to-Cognitive (Platform Orchestration)

# Ensure output directory exists
mkdir -p reports plots

echo "🚀 [1/4] Starting Physics Core: Generating 3nm Ground Truth..."
export PYTHONPATH=$PYTHONPATH:./physics_core:./cognitive_optimizer
# Using the venv from cognitive_optimizer for consistency if running from root
./cognitive_optimizer/venv/bin/python3 cognitive_optimizer/src/dummy_gen.py

echo "🧠 [2/4] Starting Cognitive Optimizer: Training FNO Surrogate..."
./cognitive_optimizer/venv/bin/python3 cognitive_optimizer/src/train.py --epochs 50

echo "🧬 [3/4] Running GEPA Evolution: Finding Optimal Config..."
./cognitive_optimizer/venv/bin/python3 cognitive_optimizer/src/gepa.py --target_loss -36.0 > reports/gepa_results.txt

echo "📊 [4/4] Generating Architectural Yield Reports..."
./cognitive_optimizer/venv/bin/python3 cognitive_optimizer/src/stats_plotter.py
./cognitive_optimizer/venv/bin/python3 cognitive_optimizer/src/signoff_reporter.py

echo "✅ Full Cycle Complete. See reports/signoff_report.md for Sign-off."
