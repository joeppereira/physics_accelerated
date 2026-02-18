import numpy as np
import torch
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

def debug_sensitivity():
    print("🕵️‍♂️ PHYSICS SENSITIVITY DIAGNOSTIC\n")
    
    bridge = OptimizerBridge()
    
    # 1. Check Temperature Spread (Area Sensitivity)
    print("1. Thermal Sensitivity Check:")
    
    # Small Area Layout
    layout_small = generate_spatial_layout(2000, 2000, 5000, dist_um=200)
    grid_small = np.zeros((16, 16))
    grid_small[layout_small == 1] = 300.0 / (np.sum(layout_small == 1))
    grid_small[layout_small == 2] = 50.0 / (np.sum(layout_small == 2))
    grid_small[layout_small == 3] = 20.0 / (np.sum(layout_small == 3))
    
    t_map_small = bridge.predict_heatmap(grid_small)
    t_rx_small = t_map_small[layout_small == 3].mean()
    
    # Large Area Layout
    layout_large = generate_spatial_layout(5000, 5000, 20000, dist_um=200)
    grid_large = np.zeros((16, 16))
    grid_large[layout_large == 1] = 300.0 / (np.sum(layout_large == 1))
    grid_large[layout_large == 2] = 50.0 / (np.sum(layout_large == 2))
    grid_large[layout_large == 3] = 20.0 / (np.sum(layout_large == 3))
    
    t_map_large = bridge.predict_heatmap(grid_large)
    t_rx_large = t_map_large[layout_large == 3].mean()
    
    print(f"   - T_rx (Small Area): {t_rx_small:.1f} C")
    print(f"   - T_rx (Large Area): {t_rx_large:.1f} C")
    print(f"   - Delta T: {t_rx_small - t_rx_large:.1f} C")
    
    if abs(t_rx_small - t_rx_large) < 5.0:
        print("   ❌ FAIL: Model predicts nearly identical temps regardless of Area.")
    else:
        print("   ✅ PASS: Significant thermal sensitivity detected.")

    # 2. Check Aging Equation Sensitivity
    print("\n2. Aging Equation Check:")
    hours = 87600.0
    
    def calc_aging(t):
        # Current formula from analyze_spatial_aging.py
        return (hours / 87600.0) * np.exp(-1000.0 / (t + 273.15))

    a_small = calc_aging(t_rx_small)
    a_large = calc_aging(t_rx_large)
    
    loss_small = 0.1 * a_small
    loss_large = 0.1 * a_large
    
    print(f"   - Aging Factor (Hot): {a_small:.4f}")
    print(f"   - Aging Factor (Cool): {a_large:.4f}")
    print(f"   - Margin Loss (Hot): {loss_small:.4f} UI")
    print(f"   - Margin Loss (Cool): {loss_large:.4f} UI")
    
    if abs(loss_small - loss_large) < 0.01:
        print("   ❌ FAIL: Aging equation is too weak to show difference.")
    else:
        print("   ✅ PASS: Significant aging difference.")

if __name__ == "__main__":
    debug_sensitivity()
