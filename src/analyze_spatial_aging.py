import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

def calculate_margin(t_rx, bias_rx, hours, dist_um, loss=-20.0):
    # 1. Base Signal
    base_eye = (bias_rx * 0.025) - (abs(loss) * 0.015)
    
    # 2. Thermal Noise (White Noise)
    noise_thermal = (t_rx - 25.0) * 0.001
    
    # 3. Crosstalk Noise (EMI coupling from TX/DSP)
    # Inverse Square Law: Noise ~ 1 / dist^2
    # Calibrated: at 50um, penalty is 0.1 UI. At 500um, penalty is 0.001 UI.
    # 0.1 = K / 50^2 -> K = 250.
    noise_xtalk = 250.0 / (dist_um**2 + 1e-9)
    
    # 4. Aging (NBTI)
    aging_factor = (hours / 87600.0) * np.exp(-3000.0 / (t_rx + 273.15))
    aging_loss = 0.5 * aging_factor
    
    margin = base_eye - noise_thermal - noise_xtalk - aging_loss
    return max(0.0, margin)

def analyze_spatial_aging():
    print("📉 Analyzing Spatial Aging: Area vs. Margin over Time...")
    bridge = OptimizerBridge()
    
    areas = np.linspace(2000, 10000, 5) 
    dist_fixed = 200.0 # Fixed for this study
    
    print("\n| Area (um^2) | T_rx (°C) | Year 0 Margin (UI) | Year 10 Margin (UI) | Degradation |")
    print("|---|---|---|---|---|")
    
    results_y0 = []
    results_y10 = []
    
    for area in areas:
        layout = generate_spatial_layout(area, area, area*3, dist_um=dist_fixed)
        
        p_dsp, p_tx, p_rx = 300.0/100.0, 50.0/10.0, 20.0/10.0
        
        power_grid = np.zeros((16, 16))
        n_dsp, n_tx, n_rx = np.sum(layout==1), np.sum(layout==2), np.sum(layout==3)
        
        if n_dsp: power_grid[layout == 1] = p_dsp / n_dsp
        if n_tx: power_grid[layout == 2] = p_tx / n_tx
        if n_rx: power_grid[layout == 3] = p_rx / n_rx
        
        temp_map = bridge.predict_heatmap(power_grid)
        t_rx = temp_map[layout == 3].mean()
        
        m0 = calculate_margin(t_rx, bias_rx=25.0, hours=0.0, dist_um=dist_fixed)
        m10 = calculate_margin(t_rx, bias_rx=25.0, hours=87600.0, dist_um=dist_fixed)
        
        results_y0.append(m0)
        results_y10.append(m10)
        
        delta = m0 - m10
        pct = (delta / m0 * 100) if m0 > 0 else 0
        
        print(f"| {area:.0f} | {t_rx:.1f} | {m0:.4f} | {m10:.4f} | -{pct:.1f}% |")

    plt.figure(figsize=(10, 6))
    plt.plot(areas, results_y0, 'o-', label='Year 0')
    plt.plot(areas, results_y10, 'x-', label='Year 10')
    plt.xlabel("Block Area (um^2)")
    plt.ylabel("Eye Margin (UI)")
    plt.title("Spatial Aging Analysis (Includes Crosstalk @ 200um)")
    plt.legend()
    plt.grid(True)
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/spatial_aging_tradeoff.png")

if __name__ == "__main__":
    analyze_spatial_aging()
