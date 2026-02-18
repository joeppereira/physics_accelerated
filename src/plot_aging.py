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
        print("❌ Model not found.")
        return
    model.eval()

    # Time sweep: 0 to 10 years
    hours = np.linspace(0, 87600, 50)
    
    # Scenario: Standard Chip (10000 um2, 10mA, K=0.3)
    # Normalized Inputs
    base_inp = [
        -0.05, -0.02, 
        10000 / NORM_FACTORS["area_um2"],
        10 / NORM_FACTORS["bias_current_ma"], 
        0.3 / NORM_FACTORS["thermal_k"],
        0.0, # Placeholder for hours
        -30 / NORM_FACTORS["loss_db"], 
        25 / NORM_FACTORS["temp_amb"]
    ]
    
    results = []
    years = hours / 8760.0
    
    report_lines = ["# 10-Year Margin Decay Report", "", "| Year | Margin (UI) | Loss vs Day 0 |", "|---|---|---|"]
    
    for i, h in enumerate(hours):
        inp = base_inp.copy()
        inp[5] = h / NORM_FACTORS["operating_hours"] # Index 5 is op_hours in schema features?
        # Check schema order:
        # FEATURES = ["ffe_m1", "ffe_p1", "area_um2", "bias_current_ma", "thermal_k", "operating_hours", "loss_db", "temp_amb"]
        # Index 0: m1, 1: p1, 2: area, 3: bias, 4: k, 5: HOURS
        
        with torch.no_grad():
            pred = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
        
        width, height, pwr, tj, life = unscale_output(pred)
        results.append(width)
        
        # Capture Year 0, 5, 10 for report
        if i == 0: day0_margin = width
        
        if abs(h - 0) < 100 or abs(h - 43800) < 1000 or abs(h - 87600) < 1000:
            loss_pct = ((day0_margin - width) / day0_margin) * 100
            report_lines.append(f"| {h/8760:.1f} | {width:.4f} UI | -{loss_pct:.1f}% |")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(years, results, 'r-', linewidth=2, label='Standard Chip (10000um^2, 10mA)')
    plt.axhline(y=0.48, color='k', linestyle='--', label='Spec Limit (0.48 UI)')
    
    plt.xlabel("Operation Time (Years)")
    plt.ylabel("Eye Width Margin (UI)")
    plt.title("SerDes Aging: Margin Decay over 10 Years")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/aging_degradation.png")
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/margin_decay_report.md", "w") as f:
        f.write("\n".join(report_lines))
        
    print("✅ Aging plot saved to plots/aging_degradation.png")
    print("✅ Report saved to reports/margin_decay_report.md")

if __name__ == "__main__":
    plot_aging_curves()