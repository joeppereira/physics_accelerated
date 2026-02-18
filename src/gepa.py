import torch
import numpy as np
import json
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

class MultiKnobOptimizer:
    def __init__(self, model_path="models/surrogate_v1.pth"):
        self.model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        except:
            print("⚠️ Warning: Model weights mismatch.")

    def optimize_spatial_layout(self, target_loss=-30.0):
        print(f"🧬 Optimizing Spatial Layout (Dist, Area) for Loss={target_loss}dB...")
        best_score = -1.0
        golden_config = {}

        for _ in range(5000):
            # Candidate: Random Placement & Sizing
            cand = {
                "area_tx_um2": np.random.uniform(1000, 5000),
                "area_rx_um2": np.random.uniform(1000, 5000),
                "area_dsp_um2": np.random.uniform(5000, 20000),
                "dist_tx_rx_um": np.random.uniform(10, 500), # The Spatial Knob
                "bias_tx_ma": 30.0,
                "bias_rx_ma": 15.0,
                "dsp_freq_ghz": 3.0,
                "thermal_k_sub": 100.0,
                "thermal_k_pkg": 5.0,
                "operating_hours": 87600.0, # EOL
                "loss_db": target_loss,
                "temp_amb": 25.0
            }

            # Scale
            inp = []
            for f in FEATURES:
                inp.append(cand.get(f, 0) / NORM_FACTORS.get(f, 1.0))

            with torch.no_grad():
                preds = self.model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
            
            # Unscale Targets
            # preds: [margin, pwr, tj_rx, tj_dsp, lifetime]
            margin = preds[0]
            tj_rx = preds[2] * NORM_FACTORS["tj_rx_c"]
            
            # Score: Maximize Margin / Total Area (Efficiency)
            total_area = cand["area_tx_um2"] + cand["area_rx_um2"] + cand["area_dsp_um2"]
            score = margin * (30000.0 / total_area)
            
            if margin > 0.1 and score > best_score:
                best_score = score
                golden_config = {
                    "dist_tx_rx": cand["dist_tx_rx_um"],
                    "area_total": total_area,
                    "eye_margin": margin,
                    "tj_rx": tj_rx
                }
        
        return golden_config

if __name__ == "__main__":
    opt = MultiKnobOptimizer()
    config = opt.optimize_spatial_layout()
    print("\n" + "="*50)
    print(f"🏆 SPATIAL GOLDEN CONFIG")
    print("="*50)
    print(f"Optimal TX-RX Dist: {config.get('dist_tx_rx', 0):.1f} um")
    print(f"Total Area        : {config.get('area_total', 0):.0f} um^2")
    print("-" * 30)
    print(f"Predicted Margin  : {config.get('eye_margin', 0):.3f} UI")
    print(f"RX Temperature    : {config.get('tj_rx', 0):.1f} °C")
    print("="*50)
