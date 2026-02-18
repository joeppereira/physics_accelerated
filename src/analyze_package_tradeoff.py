import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

def find_min_total_area(model, k_pkg_val, target_margin=0.05, target_life=10.0):
    """
    Finds the minimum Total Area required to meet specs for a given Package Quality (k_pkg).
    """
    # Sweep Total Area from 30k down to 3k
    total_areas = np.linspace(30000, 3000, 100)
    
    for area_tot in total_areas:
        # Distribute area roughly 20/20/60 %
        a_tx = area_tot * 0.2
        a_rx = area_tot * 0.2
        a_dsp = area_tot * 0.6
        
        # Fixed reasonable design
        dist = 200.0 # moderate spacing
        bias = 25.0  # high performance
        k_sub = 100.0
        
        cand = {
            "area_tx_um2": a_tx, "area_rx_um2": a_rx, "area_dsp_um2": a_dsp,
            "dist_tx_rx_um": dist,
            "bias_tx_ma": bias, "bias_rx_ma": bias/2, "dsp_freq_ghz": 3.0,
            "thermal_k_sub": k_sub, 
            "thermal_k_pkg": k_pkg_val, # THE KNOB WE ARE TESTING
            "operating_hours": 87600.0,
            "loss_db": -20.0, "temp_amb": 25.0
        }
        
        inp = [cand[f] / NORM_FACTORS.get(f, 1.0) for f in FEATURES]
        
        with torch.no_grad():
            pred = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
            
        margin, pwr, tj_rx, tj_dsp, life = unscale_output(pred)
        
        # Check Pass Criteria
        if margin >= target_margin and life >= target_life and tj_rx < 105.0:
            return area_tot, tj_rx
            
    return None, None

def analyze_packaging():
    print("📦 Analyzing Silicon Area vs. Packaging Quality...")
    
    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    try:
        model.load_state_dict(torch.load("models/surrogate_v1.pth", map_location=torch.device('cpu')))
        model.eval()
    except:
        print("❌ Model not found.")
        return

    # Sweep Package Conductivity (1 to 10)
    # 1-2: Plastic/Standard TIM
    # 4-6: Copper Lid/Good TIM
    # 8-10: Vapor Chamber/Sintered TIM
    k_values = np.linspace(1.0, 10.0, 20)
    required_areas = []
    temps = []
    
    for k in k_values:
        area, temp = find_min_total_area(model, k)
        if area:
            required_areas.append(area)
            temps.append(temp)
        else:
            required_areas.append(np.nan) # Impossible design

    # Report
    std_area = required_areas[2] # approx k=2
    adv_area = required_areas[15] # approx k=8
    savings = std_area - adv_area
    pct = (savings / std_area) * 100
    
    report = f"""# Packaging vs. Silicon Trade-off Report

## The "Heat Sink" Dividend
By improving the thermal path to ambient (better heat sink/TIM), we can significantly reduce the silicon footprint required to survive 10 years.

### Comparison
| Configuration | Package K | Description | Required Silicon Area |
|---|---|---|---|
| **Standard** | ~2.0 | Plastic/Grease | **{std_area:.0f} um²** |
| **Advanced** | ~8.0 | Vapor Chamber/Solder | **{adv_area:.0f} um²** |

### Verdict
Upgrading the cooling solution saves **{savings:.0f} um² ({pct:.1f}%)** of silicon area per lane.
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/packaging_vs_area.md", "w") as f:
        f.write(report)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, required_areas, 'o-', linewidth=2, color='purple')
    plt.xlabel("Package Conductivity K (Heat Sink Quality)")
    plt.ylabel("Min. Silicon Area Required (um^2)")
    plt.title("The Cost of Cooling: Reducing Silicon Area via Packaging")
    plt.grid(True, alpha=0.3)
    plt.annotate('Standard Pkg', xy=(2, std_area), xytext=(3, std_area+2000), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate('Advanced Pkg', xy=(8, adv_area), xytext=(6, adv_area+2000), arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.savefig("plots/packaging_tradeoff.png")
    print("✅ Plot saved: plots/packaging_tradeoff.png")
    print(f"✅ Report saved: reports/packaging_vs_area.md (Savings: {pct:.1f}%)")

if __name__ == "__main__":
    analyze_packaging()
