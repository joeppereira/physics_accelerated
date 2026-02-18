import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.bridge import OptimizerBridge
from src.schema import NORM_FACTORS

def analyze_pvt():
    print("🔬 Running Full 3D PVT Sensitivity Analysis (Process, Voltage, Temp)...")
    bridge = OptimizerBridge()
    
    p_base = np.zeros((16, 16))
    p_base[0:4, :] = 3.0 # DSP
    p_base[12:14, 2:4] = 20.0 # TX
    p_base[12:14, 6:8] = 10.0 # RX
    
    # 1. TEMPERATURE Sweep (Varying T_amb)
    ambients = np.linspace(25, 100, 10)
    res_temp = []
    for t in ambients:
        vol = bridge.predict_thermal_volume(p_base)
        res_temp.append(vol[0].max() + (t - 25.0))

    # 2. VOLTAGE Sweep (Varying Power Scaling)
    powers = np.linspace(0.5, 1.5, 10)
    res_pwr = []
    for p_scale in powers:
        vol = bridge.predict_thermal_volume(p_base * p_scale)
        res_pwr.append(vol[0].max())

    # 3. PROCESS Sweep (Varying Material K)
    # Since our FNO doesn't take K as an input channel yet, we simulate the 
    # Process variation by checking the sensitivity of the peak to Power scaling 
    # (Higher K effectively dampens the Power slope).
    # For a real Process study, we'd query the FDM solver or a conditioned FNO.
    # We'll use the FDM solver here to be physically accurate.
    print("   -> Using FDM Solver for Process (Material) Sensitivity...")
    from src.physics_engine import VoxelThermalSolver3D
    solver = VoxelThermalSolver3D(layers=5)
    k_base = [150.0, 400.0, 60.0, 10.0, 0.5]
    
    k_variations = np.linspace(0.5, 1.5, 10) # 50% to 150% conductivity
    res_proc = []
    
    # Construct Power Vol for Solver
    p_vol = np.zeros((5, 16, 16))
    p_vol[0] = p_base
    
    for k_scale in k_variations:
        k_scaled = [k * k_scale for k in k_base]
        t_vol = solver.solve(p_vol, k_scaled)
        res_proc.append(t_vol[0].max())

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    ax1.plot(ambients, res_temp, 'r-o')
    ax1.set_title("T: Ambient Sensitivity")
    ax1.set_xlabel("T_amb (°C)")
    ax1.set_ylabel("Peak Tj (°C)")
    ax1.grid(True)
    
    ax2.plot(powers, res_pwr, 'b-o')
    ax2.set_title("V: Power/Voltage Sensitivity")
    ax2.set_xlabel("Power Scale")
    ax2.grid(True)
    
    ax3.plot(k_variations, res_proc, 'g-o')
    ax3.set_title("P: Material (Process) Sensitivity")
    ax3.set_xlabel("Conductivity Scale (K)")
    ax3.grid(True)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/pvt_sensitivity_full.png")
    
    print("\n--- FINAL PVT SENSITIVITY REPORT ---")
    print(f"Corner [T]: 100C Ambient -> {res_temp[-1]:.1f}C Tj")
    print(f"Corner [V]: +50% Power    -> {res_pwr[-1]:.1f}C Tj")
    print(f"Corner [P]: -50% Cond. (K) -> {res_proc[0]:.1f}C Tj")

if __name__ == "__main__":
    analyze_pvt()