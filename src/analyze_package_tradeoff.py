import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

def find_min_total_area(model, k_pkg_val, target_margin=0.05, target_life=5.0):
    """
    Finds the minimum Total Area required.
    Target Life relaxed to 5 years to find a valid solution space.
    """
    # Sweep Total Area from 30k down to 3k
    total_areas = np.linspace(30000, 3000, 50)
    
    for area_tot in total_areas:
        a_tx = area_tot * 0.2
        a_rx = area_tot * 0.2
        a_dsp = area_tot * 0.6
        
        cand = {
            "area_tx_um2": a_tx, "area_rx_um2": a_rx, "area_dsp_um2": a_dsp,
            "dist_tx_rx_um": 300.0, # Increased distance to help
            "bias_tx_ma": 30.0, "bias_rx_ma": 15.0, "dsp_freq_ghz": 3.0,
            "thermal_k_sub": 100.0, 
            "thermal_k_pkg": k_pkg_val, 
            "operating_hours": target_life * 8760.0,
            "loss_db": -25.0, # Easier channel
            "temp_amb": 25.0
        }
        
        inp = [cand[f] / NORM_FACTORS.get(f, 1.0) for f in FEATURES]
        
        with torch.no_grad():
            pred = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
            
        margin, pwr, tj_rx, tj_dsp, life = unscale_output(pred)
        
        # Check Pass Criteria
        if margin >= target_margin and life >= target_life and tj_rx < 105.0:
            return area_tot, tj_rx
            
    return 30000.0, 105.0 # Return Max if fail (to show saturation)

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
    k_values = np.linspace(1.0, 15.0, 20) # Widen range to 15 (Liquid/Sintered)
    required_areas = []
    
    for k in k_values:
        area, temp = find_min_total_area(model, k)
        required_areas.append(area)

    # Report
    std_area = required_areas[1] # approx k=1.7
    adv_area = required_areas[10] # approx k=8.3
    
    if std_area >= 29999 and adv_area < 29000:
        # Standard saturated, Advanced found solution
        savings = std_area - adv_area
        pct = (savings / std_area) * 100
        verdict = f"Standard Packaging FAILS. Advanced saves **{savings:.0f} um²**."
    elif std_area < 29000 and adv_area < 29000:
        savings = std_area - adv_area
        pct = (savings / std_area) * 100
        verdict = f"Upgrading saves **{savings:.0f} um² ({pct:.1f}%)**."
    else:
        verdict = "Both scenarios saturated (Physics model too punitive)."
        savings = 0.0
        pct = 0.0
    
    report = f"""# Packaging vs. Silicon Trade-off Report

## The "Heat Sink" Dividend (5-Year Life, -25dB)
By improving the thermal path to ambient, we reduce the silicon area needed for thermal management.

### Comparison
| Configuration | Package K | Required Silicon Area |
|---|---|---|
| **Standard** | ~1.7 | **{std_area:.0f} um²** |
| **Advanced** | ~8.4 | **{adv_area:.0f} um²** |

### Verdict
{verdict}
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
    plt.ylim(0, 32000)
    
    plt.savefig("plots/packaging_tradeoff.png")
    print("✅ Plot saved: plots/packaging_tradeoff.png")
    print(f"✅ Report saved: reports/packaging_vs_area.md")

if __name__ == "__main__":
    analyze_packaging()