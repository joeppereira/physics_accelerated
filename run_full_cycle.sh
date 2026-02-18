#!/bin/bash
# 128G SerDes "Physics-NeMo" Design Cycle (Spatial 2D)

# Ensure output directories exist
mkdir -p reports plots data models

echo "🚀 [1/5] Starting Spatial Physics Core: Solving 2D Heat Equations..."
# Generates data/x_spatial.pt and y_spatial.pt using FDM Solver
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/python3 src/dummy_gen_normalized.py

echo "🧠 [2/5] Training 2D Fourier Neural Operator (Physics-NeMo Surrogate)..."
./venv/bin/python3 src/train.py --epochs 50

echo "🧬 [3/5] Running Spatial Placement Optimization..."
# Optimizes TX-RX distance to minimize thermal crosstalk
./venv/bin/python3 src/gepa.py > reports/spatial_optimization_results.txt

echo "📊 [4/5] Executing Multi-Physics Analysis Suite..."
# Runs the specialized trade-off studies
./venv/bin/python3 src/analyze_spatial_aging.py
./venv/bin/python3 src/analyze_isolation.py
./venv/bin/python3 src/analyze_package_tradeoff.py
./venv/bin/python3 src/analyze_neighbor_impact.py
./venv/bin/python3 src/tradeoff_plotter.py # Heatmap viz

echo "📋 [5/5] Finalizing Sign-off Report..."
./venv/bin/python3 src/signoff_reporter.py

echo "✅ Full Spatial Cycle Complete."
echo "   - View Heatmap: plots/spatial_heatmap.png"
echo "   - View Aging Plot: plots/spatial_aging_tradeoff.png"
echo "   - Read Verdict: reports/signoff_report.md"
