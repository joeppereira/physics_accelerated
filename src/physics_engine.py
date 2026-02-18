import numpy as np
from scipy.sparse import linalg, eye, kron, csr_matrix

class ThermalSolver2D:
    """
    A 2D 'Modulus-Lite' Solver for SerDes Layouts.
    Solves [G] * [T] = [P] on a 2D grid.
    Mimics the 'FPGA Heat Sink' example spatial logic.
    """
    def __init__(self, size=16, pitch_um=50):
        self.N = size
        self.dx = pitch_um
        
    def solve(self, power_grid, k_sub, k_pkg):
        """
        power_grid: (N, N) array of mW sources.
        k_sub: Lateral conductivity (Silicon).
        k_pkg: Vertical conductivity (Package/Heat Sink).
        """
        N = self.N
        # Lateral conductance (G_lat)
        g_lat = (k_sub * 1e-6) * (self.dx * 100.0) / self.dx 
        # Vertical conductance (G_vert) to Ambient (25C)
        g_vert = (k_pkg * 1e-6) * (self.dx * self.dx) / 500.0
        
        # Finite Difference 2D Laplacian Matrix
        main_diag = np.ones(N) * (4 * g_lat + g_vert)
        off_diag = np.ones(N) * (-g_lat)
        
        # 1D Block
        D = csr_matrix(np.diag(main_diag) + np.diag(off_diag[:-1], 1) + np.diag(off_diag[:-1], -1))
        # 2D Assembly
        I = eye(N)
        OFF_I = eye(N) * (-g_lat)
        G = kron(I, D) + kron(np.diag(np.ones(N-1), 1), OFF_I) + kron(np.diag(np.ones(N-1), -1), OFF_I)
        
        # Power Vector + Ambient BC
        P = power_grid.flatten() + (g_vert * 25.0)
        
        # Solve
        T_flat = linalg.spsolve(G, P)
        return T_flat.reshape((N, N))

def generate_spatial_layout(a_tx, a_rx, a_dsp, dist_um):
    """
    Creates a 16x16 grid representing the physical layout.
    """
    grid = np.zeros((16, 16))
    # Place DSP (Big source) at top
    grid[0:6, 0:16] = 1 # DSP ID
    # Place TX and RX at bottom with variable distance
    # TX at (12, 2), RX at (12, 2 + dist_nodes)
    grid[12:15, 2:5] = 2 # TX ID
    
    dist_nodes = int(dist_um / 50.0)
    rx_col = min(15, 5 + dist_nodes)
    grid[12:15, rx_col:rx_col+3] = 3 # RX ID
    
    return grid