#!/bin/bash
# 3D Voxel Architecture Upgrade

ARCHITECT_DIR="../serdes_architect"

# 1. The 3D Thermal Solver (5-Layer Stack)
cat <<EOF > "$ARCHITECT_DIR/src/thermal/solver.py"
import numpy as np
from scipy.sparse import lil_matrix, linalg

class VoxelThermalSolver3D:
    def __init__(self, size=16, layers=5, pitch_um=50, z_pitch_um=20):
        self.N = size
        self.L = layers
        self.dx = pitch_um
        self.dz = z_pitch_um
        
    def solve(self, power_vol, k_layers):
        """
        Solves 3D Poisson: ∇·(k∇T) + Q = 0
        power_vol: (L, N, N) array of power (mW) - usually only Layer 0 has power.
        k_layers: list of K values for each layer [Die, Metal, C4, Pkg, Board].
        """
        N, L = self.N, self.L
        num_voxels = L * N * N
        
        # Build Sparse Matrix (LIL is fast for construction)
        G = lil_matrix((num_voxels, num_voxels))
        P = power_vol.flatten()
        
        # Geometric Conductances
        # Lateral G = k * (dx*dz) / dx = k * dz
        # Vertical G = k * (dx*dx) / dz
        
        k_conv = 1e-3 # W/mK -> mW/umK
        
        for l in range(L):
            k = k_layers[l] * k_conv
            g_lat = k * self.dz
            g_vert = k * (self.dx**2) / self.dz
            
            for r in range(N):
                for c in range(N):
                    idx = (l * N * N) + (r * N) + c
                    
                    # 1. Lateral Neighbors (Same Layer)
                    neighbors_lat = []
                    if r > 0: neighbors_lat.append(idx - N)
                    if r < N-1: neighbors_lat.append(idx + N)
                    if c > 0: neighbors_lat.append(idx - 1)
                    if c < N-1: neighbors_lat.append(idx + 1)
                    
                    # 2. Vertical Neighbors (Up/Down Layers)
                    neighbors_vert = []
                    if l > 0: neighbors_vert.append(idx - (N*N)) # Up to Layer l-1
                    if l < L-1: neighbors_vert.append(idx + (N*N)) # Down to Layer l+1
                    
                    # 3. Build Matrix Row
                    diag_val = 0.0
                    
                    # Lateral Terms
                    for n_idx in neighbors_lat:
                        G[idx, n_idx] = -g_lat
                        diag_val += g_lat
                        
                    # Vertical Terms
                    # Note: We assume interface conductivity is harmonic mean or dominated by current layer K for simplicity
                    for n_idx in neighbors_vert:
                        G[idx, n_idx] = -g_vert
                        diag_val += g_vert
                        
                    # Boundary Condition (Top Layer = Ambient)
                    if l == L-1: # Board/Heatsink Layer
                        g_amb = g_vert * 10.0 # Efficient heat sink
                        diag_val += g_amb
                        P[idx] += g_amb * 25.0 # Ambient Ref
                        
                    G[idx, idx] = diag_val
                    
        # Solve
        try:
            T_flat = linalg.spsolve(G.tocsr(), P)
            return T_flat.reshape((L, N, N))
        except:
            return np.full((L, N, N), 25.0)
EOF

# 2. The 3D Data Generator
cat <<EOF > "$ARCHITECT_DIR/src/data_gen.py"
import torch
import numpy as np
import os
from src.thermal.solver import VoxelThermalSolver3D

def generate_nemo_dataset(samples=5000):
    print(f"🏭 Physics Factory: Generating {samples} 3D Voxel samples...")
    solver = VoxelThermalSolver3D(layers=5)
    
    x_data = [] # Input: Power Volume (L=5, 16, 16)
    y_data = [] # Output: Temp Volume (L=5, 16, 16)
    
    # Layer Properties (Die, Metal, C4, Pkg, Board)
    # K values in W/mK
    k_stack = [150.0, 400.0, 60.0, 10.0, 0.5] 
    
    for i in range(samples):
        # 1. Initialize Volume
        p_vol = np.zeros((5, 16, 16))
        
        # 2. Place Sources on DIE LAYER (Layer 0)
        # DSP (Top)
        p_dsp = np.random.uniform(50, 200)
        p_vol[0, 0:4, :] = p_dsp / (4*16)
        
        # TX (Variable)
        tx_r, tx_c = np.random.randint(10, 14), np.random.randint(1, 6)
        p_tx = np.random.uniform(50, 150)
        p_vol[0, tx_r:tx_r+2, tx_c:tx_c+2] = p_tx / 4.0
        
        # RX (Variable)
        rx_r, rx_c = np.random.randint(10, 14), np.random.randint(9, 14)
        p_rx = np.random.uniform(20, 50)
        p_vol[0, rx_r:rx_r+2, rx_c:rx_c+2] = p_rx / 4.0
        
        # 3. Solve 3D Physics
        t_vol = solver.solve(p_vol, k_stack)
        
        # 4. Normalize
        # X: Power normalized by Layer 0 density
        # Y: Temp normalized by 125C
        x_data.append(p_vol / 50.0) 
        y_data.append(t_vol / 125.0)
        
        if (i+1) % 1000 == 0:
            max_t = t_vol.max()
            print(f"   ... {i+1} solved. Peak Temp: {max_t:.1f}C (Layer {np.argmax(t_vol)//256})")

    os.makedirs("data", exist_ok=True)
    # Save as (Batch, Channels, H, W) -> (N, 5, 16, 16)
    torch.save(torch.tensor(np.array(x_data)).float(), "data/x_3d.pt")
    torch.save(torch.tensor(np.array(y_data)).float(), "data/y_3d.pt")
    print("✅ 3D Voxel Data ready.")

if __name__ == "__main__":
    generate_nemo_dataset()
EOF

echo "🚀 3D Voxel Architecture Ported."
