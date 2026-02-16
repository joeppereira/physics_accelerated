#!/bin/bash
# 128G SerDes Design Cycle: Physics-to-Cognitive (Local Orchestration)

# Ensure output directory exists
mkdir -p reports plots

# [1/4] HANDSHAKE: Generate Data
# The Physics Core acts as the "Sensor," producing grounded truth data.
# This script mixes nominal cases with "Thermal Runaway" and "Voltage Droop" failure cliffs.
echo "🚀 [1/4] Starting Physics Core: Generating 3nm Ground Truth..."
export PYTHONPATH=$PYTHONPATH:.
./venv/bin/python3 src/dummy_gen.py

# [2/4] HANDSHAKE: Sync Parquet & Train FNO
# The Cognitive Optimizer acts as the "Brain," consuming the sensor data.
# It uses Physics-Informed Loss (PIL) to learn the 35mV DFE penalty.
echo "🧠 [2/4] Starting Cognitive Optimizer: Training FNO Surrogate..."
./venv/bin/python3 src/train.py --epochs 50

# [3/4] HANDSHAKE: Evolve Config
# Using GEPA (Genetic Evolutionary Physics Accelerator) to find the Golden Configuration.
# Target Loss: -36.0 dB Channel (Simulating a long-reach backplane).
echo "🧬 [3/4] Running GEPA Evolution: Finding Optimal Config..."
./venv/bin/python3 src/gepa.py --target_loss -36.0 > reports/gepa_results.txt

# [4/4] HANDSHAKE: Sign-off
# Generates the architectural verdict, comparing AI predictions against physics rules.
echo "📊 [4/4] Generating Architectural Yield Reports..."
./venv/bin/python3 src/stats_plotter.py
./venv/bin/python3 src/signoff_reporter.py

echo "✅ Full Cycle Complete. See reports/signoff_report.md for Sign-off."