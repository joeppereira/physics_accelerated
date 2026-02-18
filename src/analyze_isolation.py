import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout
from src.analyze_spatial_aging import calculate_margin # Reuse physics logic

def analyze_isolation():
    print("↔️ Analyzing Spatial Isolation: Distance vs. Crosstalk vs. Heat...")
    bridge = OptimizerBridge()
    
    # Sweep Distance
    dists = np.linspace(50, 500, 20)
    fixed_area = 5000.0 # Medium Area
    
    margins_y0 = []
    margins_y10 = []
    temps = []
    
    for d in dists:
        layout = generate_spatial_layout(fixed_area, fixed_area, fixed_area*3, dist_um=d)
        
        p_dsp, p_tx, p_rx = 300.0/100.0, 50.0/10.0, 20.0/10.0
        
        power_grid = np.zeros((16, 16))
        n_dsp, n_tx, n_rx = np.sum(layout==1), np.sum(layout==2), np.sum(layout==3)
        
        if n_dsp: power_grid[layout == 1] = p_dsp / n_dsp
        if n_tx: power_grid[layout == 2] = p_tx / n_tx
        if n_rx: power_grid[layout == 3] = p_rx / n_rx
        
        temp_map = bridge.predict_heatmap(power_grid)
        t_rx = temp_map[layout == 3].mean()
        temps.append(t_rx)
        
        m0 = calculate_margin(t_rx, bias_rx=25.0, hours=0.0, dist_um=d)
        m10 = calculate_margin(t_rx, bias_rx=25.0, hours=87600.0, dist_um=d)
        
        margins_y0.append(m0)
        margins_y10.append(m10)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('TX-RX Distance (um)')
    ax1.set_ylabel('RX Temperature (°C)', color=color)
    ax1.plot(dists, temps, color=color, linewidth=2, label='Temp')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Eye Margin (UI)', color=color)
    ax2.plot(dists, margins_y0, color='tab:blue', linestyle='-', label='Year 0')
    ax2.plot(dists, margins_y10, color='tab:blue', linestyle='--', label='Year 10')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Isolation Optimization: Thermal & Electrical Crosstalk")
    fig.tight_layout()
    
    output = "plots/isolation_tradeoff.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output)
    print(f"✅ Isolation plot saved: {output}")
    
    # Report best distance
    best_idx = np.argmax(margins_y10)
    print(f"🏆 Optimal Distance: {dists[best_idx]:.1f} um")
    print(f"   - Max Margin (Y10): {margins_y10[best_idx]:.3f} UI")
    print(f"   - Temp at Opt: {temps[best_idx]:.1f} C")

if __name__ == "__main__":
    analyze_isolation()
