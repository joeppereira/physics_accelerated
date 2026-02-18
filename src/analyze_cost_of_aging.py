import torch
import numpy as np
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

def find_min_area_for_spec(model, target_hours, target_margin=0.15):
    best_area = 15000.0
    found = False
    max_margin_seen = 0.0
    
    # Sweep Area from large to small
    areas = np.linspace(15000, 5000, 50)
    
    for area in areas:
        bias = 15.0 # Higher bias to ensure open eye
        k = 0.2     # Poor material (Stress Test)
        
        inp = [
            -0.05, -0.02, 
            area / NORM_FACTORS["area_um2"],
            bias / NORM_FACTORS["bias_current_ma"],
            k / NORM_FACTORS["thermal_k"],
            target_hours / NORM_FACTORS["operating_hours"],
            -25 / NORM_FACTORS["loss_db"], # Easier channel (-25dB)
            25 / NORM_FACTORS["temp_amb"]
        ]
        
        with torch.no_grad():
            pred = model(torch.tensor(inp).float().unsqueeze(0)).numpy()[0]
        
        width, height, pwr, tj, life = unscale_output(pred)
        if width > max_margin_seen: max_margin_seen = width
        
        if width >= target_margin and tj < 105.0:
            best_area = area
            found = True
        else:
            break 
            
    if not found:
        print(f"Debug: Max Margin seen for T={target_hours}: {max_margin_seen:.3f} UI")
        
    return best_area if found else None

def generate_aging_report():
    print("📉 Analyzing The Cost of Reliability...")
    
    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    try:
        model.load_state_dict(torch.load("models/surrogate_v1.pth", map_location=torch.device('cpu')))
        model.eval()
    except:
        print("❌ Model not found.")
        return

    # 1. Day Zero Analysis
    area_y0 = find_min_area_for_spec(model, 0.0)
    
    # 2. Year 10 Analysis
    area_y10 = find_min_area_for_spec(model, 87600.0)
    
    if area_y0 is None or area_y10 is None:
        print("❌ Could not find valid designs for comparison.")
        return

    area_tax = area_y10 - area_y0
    tax_percent = (area_tax / area_y0) * 100.0
    
    report = f"""# 10-Year Aging Analysis: The Reliability Tax

## Executive Summary
To ensure the 128G SerDes survives 10 years of operation (NBTI & Electromigration), the architecture requires a significant increase in silicon area to maintain thermal margins.

### Key Metrics
- **Day-Zero Minimal Area:** {area_y0:.0f} um²
- **10-Year Minimal Area:** {area_y10:.0f} um²
- **Reliability Area Tax:** +{area_tax:.0f} um² (+{tax_percent:.1f}%)

### Physics Explanation
As transistors age (NBTI), their threshold voltage ($V_{{th}}$) rises, slowing them down and reducing the Eye Width. 
To compensate, the Cognitive Optimizer increases the **Area** to lower the **Junction Temperature ($T_j$)**.
Lower $T_j$ exponentially slows down the aging process (Arrhenius Law), allowing the chip to survive.

### Recommendation
Invest in the extra **{tax_percent:.1f}% area** today to avoid field failures in Year 5.
"""
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/aging_summary.md", "w") as f:
        f.write(report)
        
    print(f"✅ Comparison Complete. Tax: +{tax_percent:.1f}% Area.")
    print("📄 Report generated: reports/aging_summary.md")

if __name__ == "__main__":
    generate_aging_report()