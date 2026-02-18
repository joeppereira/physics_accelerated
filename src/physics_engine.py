import numpy as np
from scipy.sparse import lil_matrix, linalg

class VoxelThermalSolver3D:
    """
    Local Mirror of the 3D Solver in SerDes Architect.
    """
    def __init__(self, size=16, layers=5, pitch_um=50, z_pitch_um=20):
        self.N = size
        self.L = layers
        self.dx = pitch_um
        self.dz = z_pitch_um
        
    def solve(self, power_vol, k_layers):
        N, L = self.N, self.L
        num_voxels = L * N * N
        G = lil_matrix((num_voxels, num_voxels))
        P = power_vol.flatten()
        k_conv = 1e-3 
        
        for l in range(L):
            k = k_layers[l] * k_conv
            g_lat = k * self.dz
            g_vert = k * (self.dx**2) / self.dz
            
            for r in range(N):
                for c in range(N):
                    idx = (l * N * N) + (r * N) + c
                    
                    neighbors_lat = []
                    if r > 0: neighbors_lat.append(idx - N)
                    if r < N-1: neighbors_lat.append(idx + N)
                    if c > 0: neighbors_lat.append(idx - 1)
                    if c < N-1: neighbors_lat.append(idx + 1)
                    
                    neighbors_vert = []
                    if l > 0: neighbors_vert.append(idx - (N*N)) 
                    if l < L-1: neighbors_vert.append(idx + (N*N))
                    
                    diag_val = 0.0
                    
                    for n_idx in neighbors_lat:
                        G[idx, n_idx] = -g_lat
                        diag_val += g_lat
                        
                    for n_idx in neighbors_vert:
                        G[idx, n_idx] = -g_vert
                        diag_val += g_vert
                        
                    if l == L-1: 
                        g_amb = g_vert * 10.0
                        diag_val += g_amb
                        P[idx] += g_amb * 25.0
                        
                    G[idx, idx] = diag_val
                    
        try:
            T_flat = linalg.spsolve(G.tocsr(), P)
            return T_flat.reshape((L, N, N))
        except:
            return np.full((L, N, N), 25.0)

def generate_spatial_layout(a_tx, a_rx, a_dsp, dist_um):
    # Same as before, returns 2D grid which we use for Layer 0
    grid = np.zeros((16, 16))
    pixel_area = 2500.0
    
    def fill_block(start_r, start_c, area, id_val):
        num_pixels = int(area / pixel_area)
        if num_pixels < 1: num_pixels = 1
        count = 0
        r, c = start_r, start_c
        while count < num_pixels and r < 16:
            if c < 16:
                grid[r, c] = id_val
                count += 1
                c += 1
            else:
                c = start_c
                r += 1
                
    fill_block(0, 0, a_dsp, 1)
    fill_block(12, 2, a_tx, 2)
    dist_nodes = int(dist_um / 50.0)
    rx_col = min(12, 5 + dist_nodes)
    fill_block(12, rx_col, a_rx, 3)
    return grid