import torch
import numpy as np
import pandas as pd
import os
from src.surrogate import MiniSAUFNOJEPA
from src.gepa import GEPAOptimizer

def test_thermal_jitter_feedback():
    print("🧪 Verifying 'Thermal-Jitter' Feedback Loop...")
    
    # 1. Check if model exists, if not initialize a mock for the logic test
    model_path = "models/surrogate_v1.pth"
    if not os.path.exists(model_path):
        print("⚠️ No model found, using untrained surrogate for feedback check.")
        model = MiniSAUFNOJEPA()
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), model_path)
    
    # 2. Test Logic: Higher Temp -> Reduced Eye Width (The "Horizontal Tax")
    # We use the dummy data generator logic to verify the expected correlations
    from src.dummy_gen import generate
    generate() # Ensure fresh data
    
    df = pd.read_parquet("data/samples_50k.parquet")
    
    # Correlation check: Tj_c vs Eye_Width_UI
    correlation = df['tj_c'].corr(df['eye_width_ui'])
    print(f"📊 Physical Correlation (Tj vs Eye Width): {correlation:.4f}")
    
    if correlation < -0.9:
        print("✅ Thermal-Jitter Loop Pass: Strong negative correlation detected.")
    else:
        print(f"❌ Thermal-Jitter Loop Fail: Correlation {correlation} not strong enough.")

    # 3. Evolution Check: Safe-PPA Search
    print("\n🧬 Testing GEPA Optimization (Safe-PPA)...")
    optimizer = GEPAOptimizer(model_path=model_path)
    best_ffe, tj = optimizer.evolve(target_loss=-36.0)
    
    if tj < 105.0:
        print(f"✅ Safe-PPA Pass: Optimizer found stable Tj ({tj:.1f}C) for -36dB channel.")
    else:
        print(f"❌ Safe-PPA Fail: Optimizer exceeded thermal ceiling ({tj:.1f}C).")

if __name__ == "__main__":
    test_thermal_jitter_feedback()