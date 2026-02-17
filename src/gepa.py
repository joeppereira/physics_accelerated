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
            print("⚠️ Warning: Model weights mismatch or not found.")

    def find_golden_config(self, target_loss=-36.0, target_hours=87600.0):
        print(f"🧬 Searching EOL Pareto Front (Years: {target_hours/8760: .1f}, Loss: {target_loss}dB)...")
        best_ui = 0.0
        golden_config = {}

        for _ in range(5000):
            # Candidate Knobs
            raw_cand = {
                "ffe_m1": np.random.uniform(-0.12, 0),
                "ffe_p1": np.random.uniform(-0.05, 0),
                "area_um2": np.random.uniform(5000, 15000),
                "bias_current_ma": np.random.uniform(2, 20),
                "thermal_k": np.random.uniform(0.1, 0.5),
                "operating_hours": target_hours, # Fix to EOL
                "loss_db": target_loss,
                "temp_amb": 25.0
            }

            # Scale inputs
            scaled_input = [
                raw_cand["ffe_m1"],
                raw_cand["ffe_p1"],
                raw_cand["area_um2"] / NORM_FACTORS["area_um2"],
                raw_cand["bias_current_ma"] / NORM_FACTORS["bias_current_ma"],
                raw_cand["thermal_k"] / NORM_FACTORS["thermal_k"],
                raw_cand["operating_hours"] / NORM_FACTORS["operating_hours"],
                raw_cand["loss_db"] / NORM_FACTORS["loss_db"],
                raw_cand["temp_amb"] / NORM_FACTORS["temp_amb"]
            ]

            with torch.no_grad():
                preds = self.model(torch.tensor(scaled_input).float().unsqueeze(0)).numpy()[0]
            
            # Unscale output
            width_ui, height_mv, pwr_mw, tj_c, life = unscale_output(preds)

            # Optimization Criteria: Best Margin at Year 10
            if tj_c < 105.0 and width_ui > best_ui:
                best_ui = width_ui
                golden_config = {
                    "area_um2": raw_cand["area_um2"],
                    "bias_ma": raw_cand["bias_current_ma"],
                    "thermal_k": raw_cand["thermal_k"],
                    "eye_width_ui_eol": float(width_ui),
                    "tj_c": float(tj_c)
                }
        
        return golden_config

if __name__ == "__main__":
    opt = MultiKnobOptimizer()
    config = opt.find_golden_config()
    print("\n" + "="*50)
    print(f"🏆 EOL GOLDEN CONFIGURATION (10 YEAR MARGIN)")
    print("="*50)
    print(f"Area Allocation   : {config.get('area_um2', 0):.1f} um^2")
    print(f"Bias Current      : {config.get('bias_ma', 0):.2f} mA")
    print(f"Material Thermal K: {config.get('thermal_k', 0):.3f}")
    print("-" * 30)
    print(f"Predicted EOL Margin: {config.get('eye_width_ui_eol', 0):.3f} UI")
    print(f"Junction Temp       : {config.get('tj_c', 0):.1f} °C")
    print("="*50)