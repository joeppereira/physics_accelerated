import numpy as np
import matplotlib.pyplot as plt
import os
from src.physics_engine_transient import TransientThermalSolver

def power_profile(t_ms):
    """
    Defines the Power Map at time t.
    """
    p_vol = np.zeros((5, 16, 16))
    
    # Baseline DSP Load
    p_dsp = 50.0 
    
    # Burst Mode (10ms to 30ms)
    if 10.0 <= t_ms <= 30.0:
        p_dsp = 800.0 # Massive Spike
        
    # Place on grid (Layer 0, Top half)
    p_vol[0, 0:6, :] = p_dsp / (6*16)
    
    # TX/RX always active
    p_vol[0, 12:14, 2:4] = 20.0 / 4.0
    
    return p_vol

def analyze_burst():
    print("⚡ Analyzing Transient Burst Mode (Turbo Boost)...")
    
    solver = TransientThermalSolver()
    
    # Material Props
    # K (W/mK): [Die, Metal, C4, Pkg, Board]
    k_stack = [150.0, 400.0, 60.0, 10.0, 0.5]
    # Cv (J/m3K): Silicon ~1.6e6, Copper ~3.4e6, Organic ~2e6
    cv_stack = [1.6e6, 3.4e6, 2.0e6, 2.0e6, 2.0e6]
    
    times, temps = solver.solve_transient(
        power_profile, k_stack, cv_stack, duration_ms=50
    )
    
    peak_t = max(temps)
    print(f"\n🔥 Peak Transient Temp: {peak_t:.1f} °C")
    
    if peak_t > 105.0:
        print("Status: ❌ FAIL (Transient Overheat)")
    else:
        print("Status: ✅ PASS (Safe Burst)")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(times, temps, 'r-', linewidth=2)
    plt.axhline(y=105, color='k', linestyle='--', label='Limit')
    plt.axvspan(10, 30, color='orange', alpha=0.2, label='Burst Active')
    
    plt.xlabel("Time (ms)")
    plt.ylabel("Peak Die Temp (°C)")
    plt.title("Transient Thermal Response: 800mW Burst")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/transient_response.png")
    print("✅ Plot saved: plots/transient_response.png")

if __name__ == "__main__":
    analyze_burst()
