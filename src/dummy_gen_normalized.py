import pandas as pd
import numpy as np
import torch
import os
from src.physics_engine import VoxelThermalSolver3D, generate_spatial_layout

def generate_parametric_dataset(samples=1000):
    print(f"🏭 Physics Factory: Generating {samples} Parametric Material samples...")
    solver = VoxelThermalSolver3D(layers=5)
    
    x_data = [] 
    y_data = [] 
    
    for i in range(samples):
        # 1. Randomize Materials (The "Parametric" part)
        k_die = 150.0
        k_metal = np.random.uniform(200, 400) # Cu vs Al vs Ru
        k_c4 = np.random.uniform(20, 80)
        k_pkg = np.random.uniform(1, 20)      # Plastic vs Ceramic
        k_board = 0.5
        
        # Create Material Volume (L, 16, 16) - Homogeneous per layer for now
        k_vol = np.zeros((5, 16, 16))
        k_vol[0] = k_die
        k_vol[1] = k_metal
        k_vol[2] = k_c4
        k_vol[3] = k_pkg
        k_vol[4] = k_board
        
        # 2. Randomize Power (As before)
        p_vol = np.zeros((5, 16, 16))
        a_tx = np.random.uniform(1000, 5000)
        a_rx = np.random.uniform(1000, 5000)
        a_dsp = np.random.uniform(5000, 20000)
        dist = np.random.uniform(10, 500)
        layout = generate_spatial_layout(a_tx, a_rx, a_dsp, dist)
        
        p_dsp_val = np.random.uniform(50, 200)
        p_tx_val = np.random.uniform(50, 150)
        p_rx_val = np.random.uniform(20, 50)
        
        mask_dsp = (layout == 1)
        if mask_dsp.any(): p_vol[0][mask_dsp] = p_dsp_val / mask_dsp.sum()
        mask_tx = (layout == 2)
        if mask_tx.any(): p_vol[0][mask_tx] = p_tx_val / mask_tx.sum()
        mask_rx = (layout == 3)
        if mask_rx.any(): p_vol[0][mask_rx] = p_rx_val / mask_rx.sum()
        
        # 3. Solve
        t_vol = solver.solve(p_vol, k_vol)
        
        # 4. Construct Input Tensor: Stack Power and K
        # X shape: (10, 16, 16) -> First 5 are Power, Next 5 are K
        # Normalize K (Divide by 400.0 max)
        x_sample = np.concatenate([p_vol / 50.0, k_vol / 400.0], axis=0)
        
        x_data.append(x_sample)
        y_data.append(t_vol / 125.0)
        
        if (i+1) % 200 == 0:
            print(f"   ... {i+1} samples.")

    os.makedirs("data", exist_ok=True)
    torch.save(torch.tensor(np.array(x_data)).float(), "data/x_parametric.pt")
    torch.save(torch.tensor(np.array(y_data)).float(), "data/y_parametric.pt")
    print("✅ Parametric Data Ready: data/x_parametric.pt")

if __name__ == "__main__":
    generate_parametric_dataset()
