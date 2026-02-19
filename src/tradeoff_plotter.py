import numpy as np
import matplotlib.pyplot as plt
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

def plot_ai_heatmap():
    print("📉 Plotting AI-Predicted Thermal Map (64x64)...")
    bridge = OptimizerBridge()
    
    dist = 200.0
    layout = generate_spatial_layout(3000, 3000, 10000, dist)
    
    power_grid = np.zeros((64, 64))
    n_dsp = np.sum(layout == 1)
    n_tx = np.sum(layout == 2)
    n_rx = np.sum(layout == 3)
    
    if n_dsp: power_grid[layout == 1] = 300.0 / n_dsp
    if n_tx: power_grid[layout == 2] = 50.0 / n_tx
    if n_rx: power_grid[layout == 3] = 20.0 / n_rx
    
    temp_vol = bridge.predict_thermal_volume(power_grid)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(temp_vol[0], cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f"AI Thermal Map (64x64 Super-Res)\nTX-RX Dist: {dist}um")
    
    plt.savefig("plots/spatial_heatmap.png")
    print("✅ Heatmap saved to plots/spatial_heatmap.png")

if __name__ == "__main__":
    plot_ai_heatmap()