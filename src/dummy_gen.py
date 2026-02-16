import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os

def generate(samples=10000, output_path="data/samples_50k.parquet"):
    print(f"🏭 Generating {samples} physics-grounded samples (Scenario A & B enabled)...")
    
    # Base Features
    ffe_m1 = np.random.uniform(-0.1, 0, samples)
    ffe_0 = np.random.uniform(0.7, 0.9, samples)
    ffe_p1 = np.random.uniform(-0.1, 0, samples)
    ffe_p2 = np.random.uniform(-0.05, 0, samples)
    loss_db = np.random.uniform(-40, -20, samples)
    temp_ambient = np.random.uniform(25, 85, samples)
    vpp = np.random.uniform(300, 1000, samples)
    
    # Scenario A & B Mix (Simulated via distributions)
    # Scenario A: High Heat/Loss
    # Scenario B: Low Voltage/Droop
    
    # 1. Power Modeling (Derived from .lib and .itf)
    # Metal Dissipation (ITF) + Active Switching (Liberty)
    pwr_metal = 14.2 * (vpp / 800.0) # Scaled by swing
    pwr_active = 32.8 * (ffe_0 / 0.82) # Scaled by activity/ffe
    
    # 2. Thermal Loop
    tj_base = temp_ambient + ((pwr_metal + pwr_active) * 0.42)
    # Static Leakage: Exponentially scaled to Tj
    pwr_leakage = 6.4 * np.exp(0.015 * (tj_base - 25.0))
    
    tj_final = tj_base + (pwr_leakage * 0.42)
    total_pwr = pwr_metal + pwr_active + pwr_leakage

    # 3. Horizontal Margin Tax (0.01 UI per 10°C)
    thermal_tax = (tj_final - 25.0) * 0.001
    base_horizontal = 0.52 - (np.abs(loss_db) / 400.0)
    eye_width_ui = base_horizontal - thermal_tax
    
    # 4. Vertical Margin (Scenario B logic: Droop loses Gm)
    # Eye height collapses as loss increases and voltage (vpp) fluctuates
    eye_height_mv = (vpp * 0.08) - (np.abs(loss_db) * 0.6)
    
    # Efficiency
    eff_pj_bit = total_pwr / 128.0 # pJ/bit for 128G

    df = pd.DataFrame({
        "ffe_m1": ffe_m1, "ffe_0": ffe_0, "ffe_p1": ffe_p1, "ffe_p2": ffe_p2,
        "loss_db": loss_db, "temp_c": temp_ambient, "vpp": vpp,
        "pwr": total_pwr, "tj_c": tj_final,
        "eye_height_mv": eye_height_mv, "eye_width_ui": eye_width_ui,
        "pwr_leakage": pwr_leakage, "eff_pj_bit": eff_pj_bit
    })
    
    # Enforce Hard Guardrails for training
    df["status"] = np.where((df["eye_width_ui"] > 0.48) & (df["tj_c"] < 105.0), "PASS", "FAIL")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    print(f"✅ Physics data created at {output_path} (Thermal Runaway & Voltage Droop modeled)")

if __name__ == "__main__":
    generate()