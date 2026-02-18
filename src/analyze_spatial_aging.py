import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

def calculate_margin(t_rx, bias_rx, hours, loss=-20.0):
    # Base: Gain ~ Bias
    base_eye = (bias_rx * 0.025) - (abs(loss) * 0.015)
    # Thermal Noise
    noise = (t_rx - 25.0) * 0.001
    # Aging (NBTI) - Stronger K=3000
    aging_factor = (hours / 87600.0) * np.exp(-3000.0 / (t_rx + 273.15))
    aging_loss = 0.5 * aging_factor
    
    margin = base_eye - noise - aging_loss
    return max(0.0, margin)

def analyze_spatial_aging():
    print("📉 Analyzing Spatial Aging: Area vs. Margin over Time...")
    bridge = OptimizerBridge()
    
    areas = np.linspace(2000, 10000, 5) # 5 points for table
    
    print("\n| Area (um^2) | T_rx (°C) | Year 0 Margin (UI) | Year 10 Margin (UI) | Degradation |")
    print("|---|---|---|---|---|")
    
    results_y0 = []
    results_y10 = []
    
    for area in areas:
        # Layout
        layout = generate_spatial_layout(area, area, area*3, dist_um=200.0)
        
        # Power Grid (Scaled to match training distribution ~3.0 per pixel)
        # Training used p_dsp / 96. Here we simulate that density.
        p_dsp = 300.0 / 100.0 
        p_tx = 50.0 / 10.0
        p_rx = 20.0 / 10.0
        
        power_grid = np.zeros((16, 16))
        # Count pixels to calculate density
        n_dsp = np.sum(layout == 1)
        n_tx = np.sum(layout == 2)
        n_rx = np.sum(layout == 3)
        
        if n_dsp > 0: power_grid[layout == 1] = p_dsp / n_dsp
        if n_tx > 0: power_grid[layout == 2] = p_tx / n_tx
        if n_rx > 0: power_grid[layout == 3] = p_rx / n_rx
        
        # AI Inference
        temp_map = bridge.predict_heatmap(power_grid)
        t_rx = temp_map[layout == 3].mean()
        
        # Margins
        m0 = calculate_margin(t_rx, bias_rx=15.0, hours=0.0)
        m10 = calculate_margin(t_rx, bias_rx=15.0, hours=87600.0)
        
        results_y0.append(m0)
        results_y10.append(m10)
        
        delta = m0 - m10
        pct = (delta / m0 * 100) if m0 > 0 else 0
        
        print(f"| {area:.0f} | {t_rx:.1f} | {m0:.4f} | {m10:.4f} | -{pct:.1f}% |")

    # Plotting code retained for file artifact
    plt.figure(figsize=(10, 6))
    plt.plot(areas, results_y0, 'o-', label='Year 0')
    plt.plot(areas, results_y10, 'x-', label='Year 10')
    plt.xlabel("Block Area (um^2)")
    plt.ylabel("Eye Margin (UI)")
    plt.title("Spatial Aging Analysis")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/spatial_aging_tradeoff.png")

if __name__ == "__main__":
    analyze_spatial_aging()