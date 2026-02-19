import numpy as np
import torch
import os
import sys
from src.physics_engine import VoxelThermalSolver3D, generate_spatial_layout
from src.physics_engine_ir import IRDropSolver
from src.physics_engine_transient import TransientThermalSolver
from src.design_loader import DesignLoader

def test_3d_thermal_64x64():
    print("🧪 TEST 1: 3D Thermal Solver (64x64)...")
    try:
        solver = VoxelThermalSolver3D(size=64, layers=5)
        p_vol = np.zeros((5, 64, 64))
        p_vol[0, 32, 32] = 100.0 # Point source
        k_stack = [150.0, 400.0, 60.0, 10.0, 0.5]
        
        t_vol = solver.solve(p_vol, k_stack)
        peak = t_vol.max()
        print(f"   -> Peak Temp: {peak:.2f} C")
        if 25.0 < peak < 200.0:
            print("   ✅ PASS")
        else:
            print(f"   ❌ FAIL (Unrealistic Temp: {peak})")
    except Exception as e:
        print(f"   ❌ FAIL (Crash: {e})")

def test_ir_drop():
    print("\n🧪 TEST 2: IR Drop Solver...")
    try:
        solver = IRDropSolver(size=64)
        p_map = np.zeros((64, 64))
        p_map[32, 32] = 0.1 # 0.1mW load (Realistic for single node)
        t_map = np.full((64, 64), 85.0)
        
        v_map = solver.solve_ir(p_map, t_map)
        min_v = v_map.min()
        print(f"   -> Min Voltage: {min_v:.4f} V")
        if 0.5 < min_v <= 1.0:
            print("   ✅ PASS")
        else:
            print(f"   ❌ FAIL (Voltage collapse: {min_v})")
    except Exception as e:
        print(f"   ❌ FAIL (Crash: {e})")

def test_transient():
    print("\n🧪 TEST 3: Transient Burst Mode...")
    try:
        # Note: Transient solver default size is 16 in code, we should update/check
        # We will instantiate with 16 to be fast
        solver = TransientThermalSolver(size=16, dt_ms=0.001)
        k_stack = [150.0, 400.0, 60.0, 10.0, 0.5]
        cv_stack = [1.6e6, 3.4e6, 2e6, 2e6, 2e6]
        
        def p_func(t): return np.zeros((5, 16, 16))
        
        # Run small step
        times, temps = solver.solve_transient(p_func, k_stack, cv_stack, duration_ms=1.0)
        print(f"   -> Steps simulated: {len(times)}")
        print("   ✅ PASS")
    except Exception as e:
        print(f"   ❌ FAIL (Crash: {e})")

def test_design_loader():
    print("\n🧪 TEST 4: Design Loader...")
    try:
        loader = DesignLoader(grid_size=64)
        # Create dummy json
        with open("test_chip.json", "w") as f:
            f.write('{"blocks": [{"x":0,"y":0,"w":1000,"h":1000,"power_mw":100}]}')
            
        p_grid, k_layers = loader.load_from_json("test_chip.json")
        print(f"   -> Grid Shape: {p_grid.shape}")
        if p_grid.shape == (64, 64):
            print("   ✅ PASS")
        else:
            print(f"   ❌ FAIL (Shape mismatch: {p_grid.shape})")
        os.remove("test_chip.json")
    except Exception as e:
        print(f"   ❌ FAIL (Crash: {e})")

if __name__ == "__main__":
    test_3d_thermal_64x64()
    test_ir_drop()
    test_transient()
    test_design_loader()
