import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout
from src.analyze_spatial_aging import calculate_margin

def analyze_neighbor_heat():
    print("🔥 Analyzing Neighbor Impact: Ambient Temperature Sweep...")
    bridge = OptimizerBridge()
    
    # 1. Fix the Design (Standard 5000um area, 200um spacing)
    fixed_area = 5000.0
    dist = 200.0
    
    # 2. Sweep Ambient (Simulating Neighbor Load)
    # 25C (Solo) to 85C (Dense Neighbors)
    t_ambs = np.linspace(25, 85, 10)
    
    margins_y0 = []
    margins_y10 = []
    t_rx_actual = []
    
    for t_amb in t_ambs:
        layout = generate_spatial_layout(fixed_area, fixed_area, fixed_area*3, dist)
        
        # Power Grid (Scaled)
        p_dsp, p_tx, p_rx = 300.0/100.0, 50.0/10.0, 20.0/10.0
        power_grid = np.zeros((16, 16))
        n_dsp, n_tx, n_rx = np.sum(layout==1), np.sum(layout==2), np.sum(layout==3)
        if n_dsp: power_grid[layout == 1] = p_dsp / n_dsp
        if n_tx: power_grid[layout == 2] = p_tx / n_tx
        if n_rx: power_grid[layout == 3] = p_rx / n_rx
        
        # AI Prediction
        temp_map = bridge.predict_heatmap(power_grid)
        # Add (T_amb - 25) shift to the AI's delta prediction
        t_rx = temp_map[layout == 3].mean() + (t_amb - 25.0)
        t_rx_actual.append(t_rx)
        
        # Calculate Margins
        m0 = calculate_margin(t_rx, bias_rx=25.0, hours=0.0, dist_um=dist)
        m10 = calculate_margin(t_rx, bias_rx=25.0, hours=87600.0, dist_um=dist)
        
        margins_y0.append(m0)
        margins_y10.append(m10)

    # Table Report
    print("\n| Local Amb (Neighbor Heat) | T_rx Actual | Year 0 Margin | Year 10 Margin | Status |")
    print("|---|---|---|---|---|")
    for i in range(len(t_ambs)):
        status = "✅ PASS" if margins_y10[i] > 0.1 else "❌ FAIL"
        print(f"| {t_ambs[i]:.1f}°C | {t_rx_actual[i]:.1f}°C | {margins_y0[i]:.4f} | {margins_y10[i]:.4f} | {status} |")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_ambs, margins_y0, 'b-', label='Day 0')
    plt.plot(t_ambs, margins_y10, 'r--', label='Year 10 (EOL)')
    plt.axhline(y=0.1, color='k', linestyle=':', label='Spec Floor')
    plt.xlabel("Ambient Temperature (°C) - Neighbor/Global Load")
    plt.ylabel("Eye Margin (UI)")
    plt.title("Neighbor Heat Sensitivity: Global vs. Local Thermal Budget")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/neighbor_impact.png")
    print("\n✅ Plot saved: plots/neighbor_impact.png")

if __name__ == "__main__":
    analyze_neighbor_heat()
