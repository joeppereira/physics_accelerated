import numpy as np
import pandas as pd
import os
from src.physics_engine import VoxelThermalSolver3D

def analyze_corners():
    print("Corners Analysis: Simulating 7 Industry-Standard PVT Scenarios...")
    
    # Baseline Physics
    solver = VoxelThermalSolver3D(layers=5)
    k_base = np.array([150.0, 400.0, 60.0, 10.0, 0.5])
    
    # Baseline Power Grid (Layer 0)
    p_base = np.zeros((5, 16, 16))
    p_base[0, 0:4, :] = 3.0 # DSP (12mW/pixel)
    p_base[0, 12:14, 2:4] = 20.0 # TX
    p_base[0, 12:14, 6:8] = 10.0 # RX
    
    scenarios = [
        # Name, V_scale, K_scale, Load_scale
        ("Nominal", 1.00, 1.00, 1.00),
        ("FF (Fast/Hot)", 1.05, 1.20, 1.20),
        ("SS (Slow/Cool)", 0.95, 0.80, 0.80),
        ("High Load (+50%)", 1.00, 1.00, 1.50),
        ("Low Load (-50%)", 1.00, 1.00, 0.50),
        ("Voltage (+5%)", 1.05, 1.00, 1.00),
        ("Material Defect", 1.00, 0.50, 1.00)
    ]
    
    print("\n| Scenario | Power (mW) | Material K | Result Tj (°C) | Status |")
    print("|---|---|---|---|---|")
    
    results = []
    
    for name, v_scale, k_scale, load_scale in scenarios:
        # Physics Transformation
        # Power ~ Voltage^2 * Load
        # If Load is Activity, P = C * V^2 * f * alpha
        # We simplify: P_final = P_base * (V_scale**2) * Load_scale
        p_factor = (v_scale**2) * load_scale
        p_scenario = p_base * p_factor
        
        # Material Transformation
        k_scenario = k_base * k_scale
        
        # Solve
        t_vol = solver.solve(p_scenario, k_scenario)
        tj_peak = t_vol[0].max()
        
        # Check Spec
        status = "✅ PASS" if tj_peak < 105.0 else "❌ FAIL"
        
        total_pwr = np.sum(p_scenario)
        avg_k = np.mean(k_scenario)
        
        print(f"| {name:15} | {total_pwr:.1f} | {k_scale:.2f}x | {tj_peak:.1f} | {status} |")
        
        results.append({
            "Scenario": name,
            "Tj": tj_peak,
            "Power": total_pwr
        })

    # Save CSV
    df = pd.DataFrame(results)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/pvt_corners.csv", index=False)
    print(f"\n✅ Detailed report saved to reports/pvt_corners.csv")

if __name__ == "__main__":
    analyze_corners()
