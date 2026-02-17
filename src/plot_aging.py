import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

def plot_aging_curves(model_path="models/surrogate_v1.pth"):
    print("📉 Plotting Aging Degradation Curves...")
    
    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except:
        print("❌ Model not found. Run training first.")
        return
    model.eval()

    # Time sweep: 0 to 10 years
    hours = np.linspace(0, 87600, 50)
    
    # Scenario 1: Hot Chip (Small Area, Low Conductivity)
    hot_results = []
    for h in hours:
        # Inputs must be normalized!
        inp = [
            -0.05, -0.02, 
            5000 / NORM_FACTORS["area_um2"],   # Small Area
            10 / NORM_FACTORS["bias_current_ma"], 
            0.1 / NORM_FACTORS["thermal_k"],   # Poor K
            h / NORM_FACTORS["operating_hours"],
            -30 / NORM_FACTORS["loss_db"], 
            25 / NORM_FACTORS["temp_amb"]
        ]
        with torch.no_grad():
            pred = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
        # Unscale output
        width, height, pwr, tj, life = unscale_output(pred)
        hot_results.append(width)

    # Scenario 2: Cool Chip (Large Area, High Conductivity)
    cool_results = []
    for h in hours:
        inp = [
            -0.05, -0.02, 
            15000 / NORM_FACTORS["area_um2"],  # Large Area
            10 / NORM_FACTORS["bias_current_ma"], 
            0.5 / NORM_FACTORS["thermal_k"],   # Good K
            h / NORM_FACTORS["operating_hours"],
            -30 / NORM_FACTORS["loss_db"], 
            25 / NORM_FACTORS["temp_amb"]
        ]
        with torch.no_grad():
            pred = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
        width, height, pwr, tj, life = unscale_output(pred)
        cool_results.append(width)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(hours/8760, hot_results, 'r-', label='Hot Chip (Small Area, Low K)')
    plt.plot(hours/8760, cool_results, 'b-', label='Cool Chip (Large Area, High K)')
    plt.axhline(y=0.48, color='k', linestyle='--', label='Spec Limit (0.48 UI)')
    
    plt.xlabel("Operation Time (Years)")
    plt.ylabel("Eye Width Margin (UI)")
    plt.title("SerDes Aging: The Impact of Thermal Management")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output = "plots/aging_degradation.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output)
    print(f"✅ Aging plot saved to {output}")

if __name__ == "__main__":
    plot_aging_curves()
