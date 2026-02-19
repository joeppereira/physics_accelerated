import numpy as np
from scipy.sparse import lil_matrix, linalg

class VoxelThermalSolver3D:
    def __init__(self, size=16, layers=5, pitch_um=50, z_pitch_um=20):
        self.N = size
        self.L = layers
        self.dx = pitch_um
        self.dz = z_pitch_um
        
    def solve(self, power_vol, k_vol):
        """
        Solves 3D Poisson with Heterogeneous Materials.
        power_vol: (L, N, N) Power Map
        k_vol: (L, N, N) Conductivity Map (Voxel-wise K)
        """
        N, L = self.N, self.L
        num_voxels = L * N * N
        
        G = lil_matrix((num_voxels, num_voxels))
        P = power_vol.flatten()
        k_conv = 1e-3
        
        # Helper to get voxel index
        def idx(l, r, c): return (l * N * N) + (r * N) + c
        
        # Flattened K array for fast access
        K = k_vol * k_conv
        
        for l in range(L):
            for r in range(N):
                for c in range(N):
                    curr = idx(l, r, c)
                    k_curr = K[l, r, c]
                    
                    # Conductances derived from Local K
                    g_lat = k_curr * self.dz
                    g_vert = k_curr * (self.dx**2) / self.dz
                    
                    diag_sum = 0.0
                    
                    # Neighbors (6-direction)
                    # For a simplified symmetric solver, we use harmonic mean at interfaces
                    # But for speed, we stick to Nodal K dominance (valid for regular grids)
                    
                    # Lateral
                    if r > 0: 
                        n_idx = idx(l, r-1, c)
                        G[curr, n_idx] = -g_lat
                        diag_sum += g_lat
                    if r < N-1: 
                        n_idx = idx(l, r+1, c)
                        G[curr, n_idx] = -g_lat
                        diag_sum += g_lat
                    if c > 0: 
                        n_idx = idx(l, r, c-1)
                        G[curr, n_idx] = -g_lat
                        diag_sum += g_lat
                    if c < N-1: 
                        n_idx = idx(l, r, c+1)
                        G[curr, n_idx] = -g_lat
                        diag_sum += g_lat
                        
                    # Vertical
                    if l > 0:
                        n_idx = idx(l-1, r, c)
                        G[curr, n_idx] = -g_vert
                        diag_sum += g_vert
                    if l < L-1:
                        n_idx = idx(l+1, r, c)
                        G[curr, n_idx] = -g_vert
                        diag_sum += g_vert
                        
                    # Ambient BC (Top Layer)
                    if l == L-1:
                        g_amb = g_vert * 10.0
                        diag_sum += g_amb
                        P[curr] += g_amb * 25.0
                        
                    G[curr, curr] = diag_sum
                    
        try:
            T_flat = linalg.spsolve(G.tocsr(), P)
            return T_flat.reshape((L, N, N))
        except:
            return np.full((L, N, N), 25.0)

def generate_spatial_layout(a_tx, a_rx, a_dsp, dist_um):
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
