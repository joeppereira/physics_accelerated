import pandas as pd
import numpy as np

def verify_tensor_health(file_path="data/samples_normalized.parquet"):
    try:
        df = pd.read_parquet(file_path)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return

    print("🔍 Tensor Health Check:")
    for col in df.columns:
        c_min, c_max = df[col].min(), df[col].max()
        # Avoid division by zero
        denom = c_min + 1e-9 if abs(c_min) < 1e-9 else c_min
        ratio = c_max / denom if denom != 0 else 0
        
        # Check if spread is too large (unnormalized)
        # We look at absolute magnitude too. If max > 100, likely unscaled.
        is_large = abs(c_max) > 100.0
        status = "❌ UNSTABLE" if is_large else "✅ STABLE"
        
        print(f"  - {col:20}: Min={c_min:8.2f}, Max={c_max:8.2f} | {status}")

if __name__ == "__main__":
    verify_tensor_health()
