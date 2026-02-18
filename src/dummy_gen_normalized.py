import pandas as pd
import numpy as np
import torch
import os
from src.physics_engine import ThermalSolver2D, generate_spatial_layout

def generate_spatial_dataset(samples=5000):
    """
    Generates 2D Spatial datasets (Power Grids -> Temp Grids).
    This matches the NVIDIA Modulus data format.
    """
    print(f"🏭 Generating {samples} Spatial PDE solutions (16x16 Grid)...")
    size = 16
    solver = ThermalSolver2D(size=size)
    
    x_data = [] # Power Maps
    y_data = [] # Temp Maps
    
    for i in range(samples):
        # Randomize Parameters
        p_dsp = np.random.uniform(100, 500)
        p_tx = np.random.uniform(20, 100)
        p_rx = np.random.uniform(10, 50)
        dist = np.random.uniform(50, 400)
        k_pkg = np.random.uniform(1, 10)
        k_sub = 150.0
        
        # Create Power Grid
        layout = generate_spatial_layout(3000, 3000, 10000, dist)
        power_grid = np.zeros((size, size))
        power_grid[layout == 1] = p_dsp / (6*16) # Spread power over blocks
        power_grid[layout == 2] = p_tx / (3*3)
        power_grid[layout == 3] = p_rx / (3*3)
        
        # Solve PDE
        temp_grid = solver.solve(power_grid, k_sub, k_pkg)
        
        # Normalize for AI [0, 1]
        x_data.append(power_grid / 5.0) # Scale Power
        y_data.append(temp_grid / 125.0) # Scale Temp
        
        if (i+1) % 1000 == 0:
            print(f"   ... {i+1} samples solved.")

    # Save as Tensor for fast training
    os.makedirs("data", exist_ok=True)
    torch.save(torch.tensor(np.array(x_data)).float().unsqueeze(-1), "data/x_spatial.pt")
    torch.save(torch.tensor(np.array(y_data)).float().unsqueeze(-1), "data/y_spatial.pt")
    print("✅ Spatial Voxel dataset saved to data/x_spatial.pt and y_spatial.pt")

if __name__ == "__main__":
    generate_spatial_dataset()
