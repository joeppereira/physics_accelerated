import pandas as pd
import numpy as np
import torch
import os
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output
# Import the physics logic directly to test it
from src.dummy_gen_normalized import apply_aging_degradation

def debug_aging():
    print("🕵️‍♂️ AGING PHYSICS DIAGNOSTIC REPORT\n")

    # TEST 1: The Physics Equation (Ground Truth)
    print("1. Testing Physics Logic (apply_aging_degradation)...")
    base_height = 40.0
    base_width = 0.5
    temp_c = 100.0 # Hot
    
    h0 = 0.0
    h10 = 87600.0
    
    _, w0 = apply_aging_degradation(base_height, base_width, h0, temp_c)
    _, w10 = apply_aging_degradation(base_height, base_width, h10, temp_c)
    
    delta = w10 - w0
    print(f"   - Year 0 Width: {w0:.4f} UI")
    print(f"   - Year 10 Width: {w10:.4f} UI")
    print(f"   - Delta: {delta:.4f} UI")
    if delta < 0:
        print("   ✅ Physics Logic: PASS (Margin decreases with time)")
    else:
        print("   ❌ Physics Logic: FAIL (Margin increased or flat)")

    # TEST 2: Training Data Correlations
    print("\n2. Checking Training Data Correlations...")
    try:
        df = pd.read_parquet("data/samples_normalized.parquet")
        # Check correlation between normalized hours and width
        # Note: In schema, operating_hours is index 5 in FEATURES list? 
        # No, in parquet they are columns.
        
        # We need to unscale them to be sure, or just check raw correlation (sign matches).
        # operating_hours factor is positive. eye_width_ui factor is 1.0 (positive).
        
        corr = df["operating_hours"].corr(df["eye_width_ui"])
        print(f"   - Pearson Correlation (Hours vs Width): {corr:.4f}")
        
        if corr < -0.05:
            print("   ✅ Data: PASS (Negative correlation exists)")
        elif corr > 0.05:
            print("   ❌ Data: FAIL (Positive correlation - Sampling Artifact?)")
        else:
            print("   ⚠️ Data: WARNING (Correlation is very weak, hard to learn)")
            
    except Exception as e:
        print(f"   ❌ Could not read parquet: {e}")

    # TEST 3: Model Isolation Test
    print("\n3. Testing AI Model (Controlled Sweep)...")
    try:
        model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
        model.load_state_dict(torch.load("models/surrogate_v1.pth", map_location=torch.device('cpu')))
        model.eval()
        
        # Create a batch of 2 inputs: Year 0 and Year 10
        # All other inputs IDENTICAL
        # [m1, p1, area, bias, k, HOURS, loss, temp]
        base_feat = [-0.05, -0.02, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5] # Normalized values
        
        inp0 = list(base_feat)
        inp0[5] = 0.0 # Year 0
        
        inp10 = list(base_feat)
        inp10[5] = 1.0 # Year 10 (Normalized to 87600/87600 = 1.0)
        
        batch = torch.tensor([inp0, inp10]).float()
        
        with torch.no_grad():
            preds = model(batch).numpy()
            
        w_pred0 = preds[0][0] # Width is index 0
        w_pred10 = preds[1][0]
        
        print(f"   - Model Prediction Year 0: {w_pred0:.4f}")
        print(f"   - Model Prediction Year 10: {w_pred10:.4f}")
        
        if w_pred10 < w_pred0:
            print("   ✅ Model: PASS (Predicts decay)")
        else:
            print("   ❌ Model: FAIL (Predicts growth/flat)")
            
    except Exception as e:
        print(f"   ❌ Model check failed: {e}")

if __name__ == "__main__":
    debug_aging()
