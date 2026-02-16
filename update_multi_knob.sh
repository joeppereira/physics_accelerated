#!/bin/bash
# Master Update: 3nm Multi-Knob SerDes Optimization

# 1. Update the Data Schema
cat <<EOF > src/schema.py
FEATURES = ["ffe_m1", "ffe_p1", "area_um2", "bias_current_ma", "rx_impedance_ohm", "loss_db", "temp_amb"]
TARGETS = ["eye_width_ui", "eye_height_mv", "total_pwr_mw", "tj_c"]
EOF

# 2. Update the Evolution Engine with Pareto Search
cat <<EOF > src/gepa.py
import torch
import numpy as np
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS

class MultiKnobOptimizer:
    def __init__(self, model_path="models/surrogate_v1.pth"):
        # Initialize model with correct dimensions from schema
        self.model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
        except:
            print("⚠️ Warning: Model weights not found or incompatible. Initialize for structure only.")

    def find_golden_config(self, target_loss=-36.0):
        print(f"🧬 Searching Pareto Front for {target_loss}dB channel...")
        best_ui = 0
        golden_config = {}

        # Evolutionary Search (Area, Current, Impedance)
        # Search Space: Area (5k-15k), Current (2-12mA), Impedance (85-110)
        for _ in range(2000):
            # Candidate: [m1, p1, Area, Current, Zrx, Loss, Amb]
            candidate = [
                np.random.uniform(-0.12, 0), np.random.uniform(-0.05, 0),
                np.random.uniform(5000, 15000), np.random.uniform(2, 12),
                np.random.uniform(85, 110), target_loss, 25.0
            ]
            
            with torch.no_grad():
                preds = self.model(torch.tensor(candidate).float().unsqueeze(0)).numpy()[0]
                # Targets: [Width, Height, Power, Tj]
                ui, tj, pwr = preds[0], preds[3], preds[2]

            # The 3nm Constraint Gate
            if tj < 105.0 and pwr < 70.0 and ui > best_ui:
                best_ui = ui
                golden_config = {
                    "area_um2": candidate[2], "bias_ma": candidate[3],
                    "z_rx": candidate[4], "eye_width_ui": ui, "tj_c": tj,
                    "total_pwr_mw": pwr
                }
        return golden_config

if __name__ == "__main__":
    opt = MultiKnobOptimizer()
    config = opt.find_golden_config()
    print("🏆 Golden Config:", config)
EOF

# 3. Create the 3D Trade-off Visualizer
cat <<EOF > src/tradeoff_plotter.py
import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS

def generate_sweep_data(output_path="simulated_result/sweep_analysis.csv"):
    print("📊 Generating 3D Trade-off Sweep Data...")
    
    # Load Model
    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    try:
        model.load_state_dict(torch.load("models/surrogate_v1.pth", map_location=torch.device('cpu')))
    except:
        print("❌ Error: Model not trained. Run src/train.py first.")
        return

    model.eval()
    
    # Sweep Ranges
    bias_range = np.linspace(2, 12, 30) # X
    area_range = np.linspace(5000, 15000, 30) # Y
    
    results = []
    
    for bias in bias_range:
        for area in area_range:
            # Fixed params: FFE nominal, Zrx=100, Loss=-30, Temp=25
            # [m1, p1, Area, Current, Zrx, Loss, Amb]
            input_tensor = torch.tensor([
                -0.05, -0.02, area, bias, 100.0, -30.0, 25.0
            ]).float().unsqueeze(0)
            
            with torch.no_grad():
                pred = model(input_tensor).numpy()[0]
            
            # Pred: [Width, Height, Power, Tj]
            results.append({
                "bias_current_ma": bias,
                "area_um2": area,
                "eye_width_ui": pred[0],
                "tj_c": pred[3]
            })
            
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Sweep data saved to {output_path}")

def plot_3d_tradeoff(data_path="simulated_result/sweep_analysis.csv"):
    if not os.path.exists(data_path):
        generate_sweep_data(data_path)

    df = pd.read_csv(data_path)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Bias Current vs Area vs Margin
    # Z-axis is Margin (Eye Width)
    # Color is Tj
    img = ax.scatter(df['bias_current_ma'], df['area_um2'], df['eye_width_ui'], c=df['tj_c'], cmap='hot')
    
    ax.set_xlabel('Bias Current (mA)')
    ax.set_ylabel('Area (um2)')
    ax.set_zlabel('UI Margin')
    
    fig.colorbar(img, label='Junction Temp (Tj)')
    plt.title("3nm Architectural Trade-off Surface")
    
    output_path = "plots/tradeoff_3d.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output_path)
    print(f"📊 3D Trade-off Surface saved to {output_path}")

if __name__ == "__main__":
    plot_3d_tradeoff()
EOF

echo "✅ Multi-Knob Architecture successfully implemented."
