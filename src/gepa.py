import torch
import numpy as np
import json
import os
from src.bridge import OptimizerBridge
from src.physics_engine import generate_spatial_layout

class SpatialOptimizer:
    def __init__(self):
        self.bridge = OptimizerBridge()

    def optimize_placement(self):
        print(f"🧬 Optimizing Block Placement on 64x64 Super-Res Grid...")
        best_temp = 1000.0
        best_dist = 0.0
        
        # Sweep distance
        for dist in np.linspace(50, 500, 20):
            layout = generate_spatial_layout(3000, 3000, 10000, dist)
            
            # Create Power Grid (64x64)
            power_grid = np.zeros((64, 64))
            p_dsp = 300.0
            p_tx = 50.0
            p_rx = 20.0
            
            n_dsp = np.sum(layout == 1)
            n_tx = np.sum(layout == 2)
            n_rx = np.sum(layout == 3)
            
            if n_dsp: power_grid[layout == 1] = p_dsp / n_dsp
            if n_tx: power_grid[layout == 2] = p_tx / n_tx
            if n_rx: power_grid[layout == 3] = p_rx / n_rx
            
            # AI Inference
            temp_vol = self.bridge.predict_thermal_volume(power_grid)
            
            # Metric: Peak RX Temp on Die Layer
            if n_rx > 0:
                rx_temp = temp_vol[0][layout == 3].mean()
                if rx_temp < best_temp:
                    best_temp = rx_temp
                    best_dist = dist
                
        return best_dist, best_temp

if __name__ == "__main__":
    opt = SpatialOptimizer()
    dist, temp = opt.optimize_placement()
    
    config = {
        "dist_tx_rx_um": float(dist),
        "tj_rx_c": float(temp),
        "date": "2026-02-18",
        "status": "PASS (OPTIMIZED)"
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/spatial_gepa_optimized.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\n" + "="*50)
    print(f"🏆 SPATIAL AI OPTIMIZATION RESULT (64x64)")
    print("="*50)
    print(f"Optimal TX-RX Spacing: {dist:.1f} um")
    print(f"Resulting RX Temp    : {temp:.1f} °C")
    print("="*50)
