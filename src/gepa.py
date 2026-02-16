import torch
import numpy as np
import json
import os
import argparse
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

class MultiKnobOptimizer:
    def __init__(self, model_path="models/surrogate_v1.pth"):
        self.model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        except:
            print("⚠️ Warning: Model weights not found. Initialize for structure only.")

    def find_golden_config(self, target_loss=-36.0):
        print(f"🧬 Searching Pareto Front for {target_loss}dB channel...")
        best_ui = 0.0
        golden_config = {}

        # Evolutionary Search
        for _ in range(5000):
            # 1. Generate Raw Candidate
            raw_cand = {
                "ffe_m1": np.random.uniform(-0.12, 0),
                "ffe_p1": np.random.uniform(-0.05, 0),
                "area_um2": np.random.uniform(5000, 15000),
                "bias_current_ma": np.random.uniform(2, 12),
                "rx_impedance_ohm": np.random.uniform(85, 110),
                "loss_db": target_loss,
                "temp_amb": 25.0
            }

            # 2. Scale Candidate for Model
            scaled_input = [
                raw_cand["ffe_m1"], # Already small
                raw_cand["ffe_p1"],
                raw_cand["area_um2"] / NORM_FACTORS["area_um2"],
                raw_cand["bias_current_ma"] / NORM_FACTORS["bias_current_ma"],
                raw_cand["rx_impedance_ohm"] / NORM_FACTORS["rx_impedance_ohm"],
                raw_cand["loss_db"] / NORM_FACTORS["loss_db"],
                raw_cand["temp_amb"] / NORM_FACTORS["temp_amb"]
            ]

            with torch.no_grad():
                preds = self.model(torch.tensor(scaled_input).float().unsqueeze(0)).numpy()[0]
                # preds are SCALED [Width, Height, Power, Tj]
            
            # 3. Unscale Output to Physical Units
            width_ui, height_mv, pwr_mw, tj_c = unscale_output(preds)

            # 4. Check Constraints (Physical Units)
            if tj_c < 105.0 and pwr_mw < 70.0 and width_ui > best_ui:
                best_ui = width_ui
                golden_config = {
                    "area_um2": raw_cand["area_um2"],
                    "bias_ma": raw_cand["bias_current_ma"],
                    "z_rx": raw_cand["rx_impedance_ohm"],
                    "eye_width_ui": float(width_ui),
                    "tj_c": float(tj_c),
                    "total_pwr_mw": float(pwr_mw)
                }
        
        return golden_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_loss", type=float, default=-36.0)
    args = parser.parse_args()

    opt = MultiKnobOptimizer()
    config = opt.find_golden_config(target_loss=args.target_loss)
    
    # Save for reporter
    os.makedirs("reports", exist_ok=True)
    with open("reports/gepa_optimized.json", "w") as f:
        json.dump(config, f, indent=4)

    print("\n" + "="*50)
    print(f"🏆 GEPA GOLDEN CONFIGURATION (Loss: {args.target_loss}dB)")
    print("="*50)
    print(f"Area Allocation   : {config.get('area_um2', 0):.1f} um^2")
    print(f"Bias Current      : {config.get('bias_ma', 0):.2f} mA")
    print(f"RX Impedance      : {config.get('z_rx', 0):.2f} Ohm")
    print("-" * 30)
    print(f"Predicted Eye W   : {config.get('eye_width_ui', 0):.3f} UI")
    print(f"Junction Temp     : {config.get('tj_c', 0):.1f} °C")
    print(f"Total Power       : {config.get('total_pwr_mw', 0):.1f} mW")
    print("="*50)