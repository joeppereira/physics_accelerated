import numpy as np
from scipy.sparse import linalg, eye, kron, csr_matrix

class ThermalSolver2D:
    def __init__(self, size=16, pitch_um=50):
        self.N = size
        self.dx = pitch_um
        
    def solve(self, power_grid, k_sub, k_pkg):
        N = self.N
        g_lat = (k_sub * 1e-6) * (self.dx * 100.0) / self.dx 
        g_vert = (k_pkg * 1e-6) * (self.dx * self.dx) / 500.0
        
        main_diag = np.ones(N) * (4 * g_lat + g_vert)
        off_diag = np.ones(N) * (-g_lat)
        
        D = csr_matrix(np.diag(main_diag) + np.diag(off_diag[:-1], 1) + np.diag(off_diag[:-1], -1))
        I = eye(N)
        OFF_I = eye(N) * (-g_lat)
        G = kron(I, D) + kron(np.diag(np.ones(N-1), 1), OFF_I) + kron(np.diag(np.ones(N-1), -1), OFF_I)
        
        P = power_grid.flatten() + (g_vert * 25.0)
        T_flat = linalg.spsolve(G, P)
        return T_flat.reshape((N, N))

def generate_spatial_layout(a_tx, a_rx, a_dsp, dist_um):
    """
    Creates a 16x16 grid with DYNAMIC block sizes.
    a_tx, a_rx, a_dsp in um^2.
    Pixel area = 50*50 = 2500 um^2.
    """
    grid = np.zeros((16, 16))
    pixel_area = 2500.0
    
    # Calculate dimensions in pixels
    # Assume rectangular shapes width roughly 2x height
    
    def fill_block(start_r, start_c, area, id_val):
        num_pixels = int(area / pixel_area)
        if num_pixels < 1: num_pixels = 1
        
        # Simple filling: fill rows
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
                
    # Place DSP (Top)
    fill_block(0, 0, a_dsp, 1)
    
    # Place TX (Bottom Left)
    fill_block(12, 2, a_tx, 2)
    
    # Place RX (Bottom Right, separated by dist)
    dist_nodes = int(dist_um / 50.0)
    rx_col = min(12, 5 + dist_nodes) # Ensure it fits
    fill_block(12, rx_col, a_rx, 3)
    
    return grid
