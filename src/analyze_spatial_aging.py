import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

def calculate_margin(t_rx, bias_rx, hours, loss=-30.0):
    # 1. Physics Base
    # Gain ~ Bias
    base_eye = (bias_rx * 0.025) - (abs(loss) * 0.015)
    
    # 2. Thermal Noise
    noise = (t_rx - 25.0) * 0.001
    
    # 3. Aging (NBTI)
    # Arrhenius acceleration (Stronger K=3000)
    aging_factor = (hours / 87600.0) * np.exp(-3000.0 / (t_rx + 273.15))
    # Penalty magnitude boosted to 0.5 UI max
    aging_loss = 0.5 * aging_factor
    
    margin = base_eye - noise - aging_loss
    return max(0.0, margin)

def analyze_spatial_aging():
    print("📉 Analyzing Spatial Aging: Area vs. Margin over Time...")
    bridge = OptimizerBridge()
    
    # Sweep Area (via block size in grid nodes?)
    # Layout generator takes area in um2.
    areas = np.linspace(2000, 10000, 10) # Area per block
    hours = np.linspace(0, 87600, 5) # 0, 2.5, 5, 7.5, 10 Years
    
    results = {h: [] for h in hours}
    
    for area in areas:
        # Generate Layout (Fixed Dist=200um)
        layout = generate_spatial_layout(area, area, area*3, dist_um=200.0)
        
        # Create Power Grid
        # Power density const? Or Total Power const?
        # Usually larger area = lower density -> cooler.
        # Let's assume Fixed Power (Optimization scenario: Spread the heat).
        p_dsp = 300.0
        p_tx = 50.0
        p_rx = 20.0
        
        power_grid = np.zeros((16, 16))
        power_grid[layout == 1] = p_dsp / np.sum(layout == 1)
        power_grid[layout == 2] = p_tx / np.sum(layout == 2)
        power_grid[layout == 3] = p_rx / np.sum(layout == 3)
        
        # AI Inference (Get Temp Map)
        temp_map = bridge.predict_heatmap(power_grid)
        
        # Extract RX Temp
        t_rx = temp_map[layout == 3].mean()
        
        # Calculate Margins over time
        for h in hours:
            m = calculate_margin(t_rx, bias_rx=15.0, hours=h)
            results[h].append(m)

    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(hours)))
    
    for i, h in enumerate(hours):
        year = h / 8760.0
        plt.plot(areas, results[h], 'o-', color=colors[i], label=f'Year {year:.1f}')
        
    plt.xlabel("Block Area (um^2) - Heat Spreading")
    plt.ylabel("Eye Margin (UI)")
    plt.title("Spatial Aging Analysis: The Benefit of Area")
    plt.axhline(y=0.1, color='r', linestyle='--', label='Fail Limit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output = "plots/spatial_aging_tradeoff.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output)
    print(f"✅ Plot saved: {output}")

if __name__ == "__main__":
    analyze_spatial_aging()
