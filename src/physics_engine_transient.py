import numpy as np
from scipy.sparse import lil_matrix, eye, kron, csr_matrix

class TransientThermalSolver:
    def __init__(self, size=16, layers=5, pitch_um=50, z_pitch_um=20, dt_ms=0.001):
        self.N = size
        self.L = layers
        self.dx = pitch_um
        self.dz = z_pitch_um
        self.dt = dt_ms * 1e-3 # Seconds
        
    def solve_transient(self, power_vol_func, k_layers, cv_layers, duration_ms=100):
        """
        Solves Time-Dependent Heat Equation.
        power_vol_func: Function(time_ms) -> (L, N, N) Power Map.
        cv_layers: Heat Capacity (J/m^3K) per layer.
        """
        N, L = self.N, self.L
        num_voxels = L * N * N
        
        # 1. Build Conductance Matrix (G) - Same as Steady State
        # We reuse the logic but kept optimized for update loop
        G = self._build_conductance_matrix(k_layers)
        
        # 2. Build Capacitance Matrix (C)
        # C_node = Vol * Cv = (dx*dx*dz) * Cv (units?)
        # Unit check: Power in mW. G in mW/K. 
        # Heat Cap in J/K? 
        # J = W*s = mW * ms? No, J = W*s.
        # Cv (J/m^3K). Vol (um^3).
        # Cap_J = Cv * Vol * 1e-18.
        # Cap_mWs = Cap_J * 1000.
        
        vol_m3 = (self.dx * 1e-6)**2 * (self.dz * 1e-6)
        C_vec = np.zeros(num_voxels)
        
        for l in range(L):
            cv = cv_layers[l]
            cap_val = cv * vol_m3 * 1000.0 # mWs/K (mJ/K)
            idx_start = l * N * N
            idx_end = (l+1) * N * N
            C_vec[idx_start:idx_end] = cap_val
            
        # 3. Time Stepping (Explicit Euler)
        # T_new = T_old + (dt/C) * (P_in - G*T_old)
        
        T = np.full(num_voxels, 25.0) # Initial Condition
        
        steps = int(duration_ms / (self.dt * 1000))
        history = []
        times = []
        
        # Pre-compute Ambient flux vector (constant part of G*T)
        # Actually G includes G_amb. P_in should include G_amb*T_amb?
        # In my G matrix, I usually put G_amb in diagonal. 
        # Net Flow = P_in - G*T.
        # If G*T accounts for flow TO ambient, P_in must NOT include it?
        # Wait, Flow = G*(T - Tamb). = G*T - G*Tamb.
        # So Heat_In = P_source - (G*T - G*Tamb) = (P_source + G*Tamb) - G*T.
        # Let's verify _build_G handles G_amb.
        
        print(f"⏱️ Simulating {duration_ms}ms in {steps} steps...")
        
        for step in range(steps):
            t_ms = step * self.dt * 1000
            
            # Get Instantaneous Power
            P_source = power_vol_func(t_ms).flatten()
            
            # Add Ambient Injection (G_amb * 25.0)
            # We need the G_amb vector from build_G
            # For simplicity, let's assume _build_G returns the full G matrix 
            # and we calculate flow G*T.
            # But we need the offset vector.
            
            # Refined update: T += (dt/C) * (P_source + P_amb_offset - G @ T)
            
            flux = P_source + self.P_amb_offset - (G @ T)
            T = T + (self.dt / C_vec) * flux
            
            # Record Peak Temp
            if step % 10 == 0:
                history.append(T.max())
                times.append(t_ms)
                
        return times, history

    def _build_conductance_matrix(self, k_layers):
        N, L = self.N, self.L
        num_voxels = L * N * N
        G = lil_matrix((num_voxels, num_voxels))
        k_conv = 1e-3
        self.P_amb_offset = np.zeros(num_voxels)
        
        for l in range(L):
            k = k_layers[l] * k_conv
            g_lat = k * self.dz
            g_vert = k * (self.dx**2) / self.dz
            
            for r in range(N):
                for c in range(N):
                    idx = (l * N * N) + (r * N) + c
                    diag_val = 0.0
                    
                    # Neighbors logic (simplified for brevity, assume same as before)
                    # For Transient, G matrix is just conductance.
                    
                    # Lat
                    if r>0: G[idx, idx-N] = -g_lat; diag_val += g_lat
                    if r<N-1: G[idx, idx+N] = -g_lat; diag_val += g_lat
                    if c>0: G[idx, idx-1] = -g_lat; diag_val += g_lat
                    if c<N-1: G[idx, idx+1] = -g_lat; diag_val += g_lat
                    
                    # Vert
                    if l>0: G[idx, idx-N*N] = -g_vert; diag_val += g_vert
                    if l<L-1: G[idx, idx+N*N] = -g_vert; diag_val += g_vert
                    
                    # Ambient
                    if l==L-1:
                        g_amb = g_vert * 10.0
                        diag_val += g_amb
                        self.P_amb_offset[idx] = g_amb * 25.0
                        
                    G[idx, idx] = diag_val
                    
        return G.tocsr()
