import numpy as np
import matplotlib.pyplot as plt
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

def plot_ai_heatmap():
    print("📉 Plotting AI-Predicted Thermal Map...")
    bridge = OptimizerBridge()
    
    # Generate a layout
    dist = 200.0
    layout = generate_spatial_layout(3000, 3000, 10000, dist)
    
    # Create Power Grid
    power_grid = np.zeros((16, 16))
    power_grid[layout == 1] = 300.0 / (6*16) # DSP
    power_grid[layout == 2] = 50.0 / (3*3)   # TX
    power_grid[layout == 3] = 20.0 / (3*3)   # RX
    
    # Predict
    temp_map = bridge.predict_heatmap(power_grid)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(temp_map, cmap='inferno', interpolation='bicubic')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f"AI-Predicted Heat Map (2D FNO)\nTX-RX Dist: {dist}um")
    
    # Annotate blocks roughly
    plt.text(8, 3, "DSP Core", color='white', ha='center')
    plt.text(3, 13, "TX", color='white', ha='center')
    plt.text(3 + int(dist/50), 13, "RX", color='white', ha='center')
    
    output = "plots/spatial_heatmap.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output)
    print(f"✅ Heatmap saved to {output}")

if __name__ == "__main__":
    plot_ai_heatmap()
