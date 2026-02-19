import numpy as np
from scipy.sparse import lil_matrix, linalg

class IRDropSolver:
    """
    Solves 2D Electrical Grid for Voltage Drop (IR Drop).
    Coupled with Thermal: R increases with T.
    """
    def __init__(self, size=64, pitch_um=31.25):
        self.N = size
        self.dx = pitch_um
        self.vdd = 1.0 # Nominal Voltage
        
    def solve_ir(self, power_map, temp_map, sheet_res=0.1):
        """
        Solves [G] * [V] = [I]
        power_map: (N,N) Power consumption (mW) -> Source Current I = P/V
        temp_map: (N,N) Temperature (C) -> Affects Resistance
        sheet_res: Ohm/square at 25C.
        """
        N = self.N
        num_nodes = N * N
        
        G_mat = lil_matrix((num_nodes, num_nodes))
        I_vec = np.zeros(num_nodes)
        
        # Temp Coefficient for Copper (0.004 per C)
        alpha = 0.004
        
        for r in range(N):
            for c in range(N):
                idx = r * N + c
                
                # Local Temperature effect
                t_local = temp_map[r, c]
                rho_scale = 1.0 + alpha * (t_local - 25.0)
                
                # Effective Conductance between nodes
                # G = 1 / (R_sheet * rho_scale)
                g_local = 1.0 / (sheet_res * rho_scale)
                
                diag_sum = 0.0
                
                # 4 Neighbors (Resistor Mesh)
                neighbors = []
                if r > 0: neighbors.append(idx - N)
                if r < N-1: neighbors.append(idx + N)
                if c > 0: neighbors.append(idx - 1)
                if c < N-1: neighbors.append(idx + 1)
                
                for n_idx in neighbors:
                    G_mat[idx, n_idx] = -g_local
                    diag_sum += g_local
                    
                # Power Supply Connection (VDD)
                # Assume VDD is connected at edges (Power Ring)
                # G_vdd connects node to Ideal 1.0V source
                if r == 0 or r == N-1 or c == 0 or c == N-1:
                    g_vdd = 10.0 # Strong connection
                    diag_sum += g_vdd
                    I_vec[idx] += g_vdd * self.vdd
                
                G_mat[idx, idx] = diag_sum
                
                # Current Sink (Load)
                # I = P / V (Approximation: use V_nominal)
                p_load = power_map[r, c]
                i_load = p_load / self.vdd # mA
                I_vec[idx] -= i_load # Current LEAVING node
                
        # Solve for V
        try:
            V_map = linalg.spsolve(G_mat.tocsr(), I_vec)
            return V_map.reshape((N, N))
        except:
            return np.full((N, N), self.vdd)

if __name__ == "__main__":
    # Quick Test
    solver = IRDropSolver(size=64)
    p_map = np.zeros((64, 64))
    p_map[32, 32] = 100.0 # Hotspot current sink
    t_map = np.full((64, 64), 85.0)
    
    v_map = solver.solve_ir(p_map, t_map)
    print(f"Min Voltage: {v_map.min():.4f} V (Drop: {(1.0-v_map.min())*1000:.1f} mV)")
