import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from src.schema import NORM_FACTORS

def generate_normalized_samples(samples=50000):
    print(f"🏭 Generating {samples} pre-scaled, physically-consistent samples...")
    
    # 1. Generate Raw Physical Inputs
    area = np.random.uniform(5000, 15000, samples)
    bias = np.random.uniform(2, 12, samples)
    z_rx = np.random.uniform(85, 110, samples)
    loss = np.random.uniform(-40, -20, samples)
    temp_amb = np.random.uniform(25, 85, samples)
    
    # 2. Apply "Physics Engine" logic (The Handshake)
    # Power = Static (leakage) + Dynamic (bias * voltage)
    pwr = 10 + (bias * 0.75 * 64 * 0.5) # Simplified 3nm power model
    # Tj = Tamb + (Power * Rth_ja) | Rth_ja improves with Area
    r_theta = 0.42 * np.sqrt(10000 / area)
    tj = temp_amb + (pwr * r_theta)
    
    # Eye Margins with Thermal Tax
    # EyeHeight scales with bias and inversely with loss
    height = (bias * 5.0) - (abs(loss) * 0.8)
    # EyeWidth (UI) has the 0.01 UI per 10C tax
    width = 0.6 - ((tj - 25.0) / 10.0) * 0.01

    # 3. CONSTRUCT NORMALIZED DATAFRAME
    df = pd.DataFrame({
        "ffe_m1": np.random.uniform(-0.12, 0, samples),
        "ffe_p1": np.random.uniform(-0.05, 0, samples),
        "area_um2": area / NORM_FACTORS["area_um2"],
        "bias_current_ma": bias / NORM_FACTORS["bias_current_ma"],
        "rx_impedance_ohm": z_rx / NORM_FACTORS["rx_impedance_ohm"],
        "loss_db": abs(loss) / 40.0, # Scale -40 to 1.0
        "temp_amb": temp_amb / 125.0,
        # Targets
        "eye_width_ui": width, # Already ~[0.4, 0.6]
        "eye_height_mv": height / NORM_FACTORS["eye_height_mv"],
        "total_pwr_mw": pwr / NORM_FACTORS["total_pwr_mw"],
        "tj_c": tj / NORM_FACTORS["tj_c"]
    })

    os.makedirs("data", exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, "data/samples_normalized.parquet")
    print(f"✅ Generated {samples} normalized samples for stable training.")

if __name__ == "__main__":
    generate_normalized_samples()
