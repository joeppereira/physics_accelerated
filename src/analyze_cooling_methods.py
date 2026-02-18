import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout
from src.schema import NORM_FACTORS, unscale_output

def calculate_margin(t_rx, bias_rx, hours, loss=-20.0):
    # Same physics logic as before
    base_eye = (bias_rx * 0.025) - (abs(loss) * 0.015)
    noise = (t_rx - 25.0) * 0.001
    aging_factor = (hours / 87600.0) * np.exp(-3000.0 / (t_rx + 273.15))
    aging_loss = 0.5 * aging_factor
    return max(0.0, base_eye - noise - aging_loss)

def analyze_cooling_tradeoff():
    print("❄️ Analyzing Alternative Cooling Methods (Fixed Small Area)...")
    bridge = OptimizerBridge()
    
    # 1. FIX THE SILICON (Small, Cheap Chip)
    # Total Area ~9000 (3k + 3k + 3k) is small for this power.
    # Increase to 5000 (challenging but possible)
    fixed_area = 5000.0 
    dist = 200.0
    
    # 2. SWEEP THE COOLING (The "Alternative Methods")
    # Y-Axis: Packaging Quality (K_pkg)
    k_vals = np.linspace(1.0, 20.0, 20) 
    # X-Axis: Ambient Temperature (Active Cooling)
    t_ambs = np.linspace(85.0, 20.0, 20) # 85=Passive, 20=Liquid/Chiller
    
    # Grid for Heatmap
    results_margin = np.zeros((len(k_vals), len(t_ambs)))
    
    # Generate Layout once (Geometry is fixed)
    layout = generate_spatial_layout(fixed_area, fixed_area, fixed_area*3, dist)
    
    # Power Grid (Fixed Power)
    p_dsp, p_tx, p_rx = 300.0, 50.0, 20.0
    # Scaled densities for model
    power_grid_base = np.zeros((16, 16))
    power_grid_base[layout == 1] = p_dsp / 100.0
    power_grid_base[layout == 2] = p_tx / 10.0
    power_grid_base[layout == 3] = p_rx / 10.0
    
    # We must Manually "Simulate" the K_pkg and T_amb effect 
    # because the FNO was trained with specific K_pkg/T_amb ranges.
    # Wait, the FNO takes K_pkg and T_amb as inputs?
    # NO. The FNO takes `power_grid`.
    # The `dummy_gen` BAKED K_pkg and T_amb into the `y` (Temp Grid).
    # The FNO I trained maps P -> T.
    # Does it take K_pkg and T_amb as *inputs*?
    # Let's check `src/surrogate.py`: `self.fc0 = nn.Linear(1, self.width)`.
    # It ONLY takes `x` (Power Grid) channel!
    
    # CRITICAL REALIZATION:
    # My current FNO is "Material-Agnostic" (or rather, averaged). 
    # It does NOT take K_pkg as an input channel. 
    # It learned the average physics of the training set.
    
    # To simulate K_pkg effects *with this specific model*, I must rely on the 
    # Physics Engine directly (The Teacher), OR retrain a "Parametric FNO" (Conditioned FNO).
    
    # Since we want accurate results fast, and we have the `PhysicsEngine` locally,
    # we should use `ThermalSolver2D` directly for this study. 
    # The FNO is a fast surrogate, but if it lacks the K_pkg input, it can't predict K_pkg changes.
    
    print("   -> Using Physics Engine (FDM) for high-fidelity material sweep...")
    from src.physics_engine import ThermalSolver2D
    
    solver = ThermalSolver2D(size=16)
    
    # Actual Power Grid (mW) for Solver
    phys_power_grid = np.zeros((16, 16))
    # Correct normalization for solver (mW per node)
    # The solver takes raw mW array.
    # In generate_spatial_layout, we fill blocks.
    # We need to distribute 300mW over the DSP pixels.
    n_dsp = np.sum(layout == 1)
    n_tx = np.sum(layout == 2)
    n_rx = np.sum(layout == 3)
    
    phys_power_grid[layout == 1] = p_dsp / n_dsp
    phys_power_grid[layout == 2] = p_tx / n_tx
    phys_power_grid[layout == 3] = p_rx / n_rx
    
    k_sub = 150.0 # Silicon
    
    for i, k in enumerate(k_vals):
        for j, t_amb in enumerate(t_ambs):
            # Solve Physics
            temp_grid = solver.solve(phys_power_grid, k_sub, k)
            
            # Adjust for Ambient (Solver output is Delta T + 25 reference?)
            # Solver `P = power + g_vert * 25.0`. It assumes ambient 25.
            # If we change ambient, we change the boundary condition P vector.
            # But `solver.solve` hardcodes `25.0`.
            # Linear correction: T_new = T_solved - 25.0 + T_new_amb.
            t_map = temp_grid - 25.0 + t_amb
            
            t_rx = t_map[layout == 3].mean()
            
            # Calc Margin at Year 10
            m10 = calculate_margin(t_rx, bias_rx=25.0, hours=87600.0, loss=-20.0)
            results_margin[i, j] = m10

    # Plot Heatmap
    plt.figure(figsize=(10, 8))
    # Extent: [Left, Right, Bottom, Top] -> [T_max, T_min, K_min, K_max]
    # T_ambs is 85..20 (descending index?). Let's flip for plot.
    
    plt.imshow(results_margin, aspect='auto', cmap='RdYlGn', origin='lower',
               extent=[85, 20, 1, 20])
    
    plt.colorbar(label='Year 10 Eye Margin (UI)')
    plt.xlabel("Ambient Temperature (°C) - Active Cooling")
    plt.ylabel("Package Conductivity (W/mK) - Heat Sink")
    plt.title(f"Alternative Cooling Trade-offs\n(Fixed Small Area: {fixed_area} um^2)")
    
    # Contours for Spec
    CS = plt.contour(t_ambs, k_vals, results_margin, levels=[0.1, 0.2, 0.3], colors='white')
    plt.clabel(CS, inline=1, fontsize=10)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/cooling_tradeoff.png")
    print("✅ Plot saved: plots/cooling_tradeoff.png")
    
    # Text Report
    # Find points that pass Margin > 0.25
    print("\n| Cooling Solution | T_amb | K_pkg | Result (Margin) |")
    print("|---|---|---|---|")
    scenarios = [
        ("Passive / Plastic", 85, 2),
        ("Fan / Copper", 45, 8),
        ("Liquid / Sintered", 25, 15)
    ]
    
    for name, t, k in scenarios:
        # Find nearest indices
        idx_t = (np.abs(t_ambs - t)).argmin()
        idx_k = (np.abs(k_vals - k)).argmin()
        res = results_margin[idx_k, idx_t]
        status = "✅ PASS" if res > 0.25 else "❌ FAIL"
        print(f"| {name} | {t}C | {k} | {res:.3f} UI {status} |")

if __name__ == "__main__":
    analyze_cooling_tradeoff()
