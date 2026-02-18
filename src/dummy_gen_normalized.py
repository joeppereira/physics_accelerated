import pandas as pd
import numpy as np
import torch
import os
from src.physics_engine import VoxelThermalSolver3D, generate_spatial_layout
from src.schema import scale_data

def generate_3d_dataset(samples=1000):
    print(f"🏭 Generating {samples} 3D Voxel samples (Local)...")
    solver = VoxelThermalSolver3D(layers=5)
    
    x_data = [] 
    y_data = [] 
    
    k_stack = [150.0, 400.0, 60.0, 10.0, 0.5]
    
    for i in range(samples):
        # 1. Initialize Volume
        p_vol = np.zeros((5, 16, 16))
        
        # 2. Layout (Layer 0)
        # Variable geometry
        a_tx = np.random.uniform(1000, 5000)
        a_rx = np.random.uniform(1000, 5000)
        a_dsp = np.random.uniform(5000, 20000)
        dist = np.random.uniform(10, 500)
        
        layout = generate_spatial_layout(a_tx, a_rx, a_dsp, dist)
        
        # Powers
        p_dsp = np.random.uniform(50, 200)
        p_tx = np.random.uniform(50, 150)
        p_rx = np.random.uniform(20, 50)
        
        # Map Layout to Power Volume
        # DSP (ID 1)
        mask_dsp = (layout == 1)
        if mask_dsp.any(): p_vol[0][mask_dsp] = p_dsp / mask_dsp.sum()
            
        # TX (ID 2)
        mask_tx = (layout == 2)
        if mask_tx.any(): p_vol[0][mask_tx] = p_tx / mask_tx.sum()
            
        # RX (ID 3)
        mask_rx = (layout == 3)
        if mask_rx.any(): p_vol[0][mask_rx] = p_rx / mask_rx.sum()
        
        # 3. Solve
        t_vol = solver.solve(p_vol, k_stack)
        
        # 4. Store (Normalized)
        x_data.append(p_vol / 50.0)
        y_data.append(t_vol / 125.0)
        
        if (i+1) % 200 == 0:
            print(f"   ... {i+1} samples.")

    os.makedirs("data", exist_ok=True)
    # Save as (Batch, Channels, H, W)
    torch.save(torch.tensor(np.array(x_data)).float(), "data/x_3d.pt")
    torch.save(torch.tensor(np.array(y_data)).float(), "data/y_3d.pt")
    print("✅ Local 3D Voxel Data ready.")

if __name__ == "__main__":
    generate_3d_dataset()