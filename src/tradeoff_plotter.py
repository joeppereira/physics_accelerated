import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS

def plot_spatial_tradeoff():
    print("📉 Plotting Spatial Thermal Crosstalk...")
    
    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    try:
        model.load_state_dict(torch.load("models/surrogate_v1.pth", map_location=torch.device('cpu')))
        model.eval()
    except:
        return

    # Sweep Distance
    dists = np.linspace(10, 500, 50)
    margins = []
    temps = []
    
    for d in dists:
        cand = {
            "area_tx_um2": 3000, "area_rx_um2": 3000, "area_dsp_um2": 10000,
            "dist_tx_rx_um": d,
            "bias_tx_ma": 30.0, "bias_rx_ma": 15.0, "dsp_freq_ghz": 3.0,
            "thermal_k_sub": 100.0, "thermal_k_pkg": 5.0,
            "operating_hours": 87600.0,
            "loss_db": -30.0, "temp_amb": 25.0
        }
        
        inp = [cand[f] / NORM_FACTORS.get(f, 1.0) for f in FEATURES]
        with torch.no_grad():
            preds = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
            
        margins.append(preds[0]) # Margin
        temps.append(preds[2] * NORM_FACTORS["tj_rx_c"]) # Tj RX

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('TX-RX Spacing (um)')
    ax1.set_ylabel('RX Temperature (°C)', color=color)
    ax1.plot(dists, temps, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Eye Margin (UI)', color=color)
    ax2.plot(dists, margins, color=color, linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Spatial Optimization: Thermal Crosstalk vs. Performance")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/spatial_tradeoff.png")
    print("✅ Plot saved: plots/spatial_tradeoff.png")

if __name__ == "__main__":
    plot_spatial_tradeoff()