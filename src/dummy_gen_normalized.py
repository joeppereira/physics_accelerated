import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from src.schema import NORM_FACTORS, scale_data

def apply_aging_degradation(eye_height, eye_width, hours, temp_c):
    aging_factor = (hours / 87600.0) * np.exp(-1000.0 / (temp_c + 273.15 + 1e-9))
    eol_height = eye_height * (1.0 - (0.15 * aging_factor)) 
    eol_width = eye_width - (0.05 * aging_factor)
    return eol_height, eol_width

def generate_normalized_samples(samples=50000):
    print(f"🏭 Generating {samples} samples with MATERIAL & AGING PHYSICS...")
    
    # 1. Inputs (Physical Units)
    area = np.random.uniform(5000, 15000, samples)
    bias = np.random.uniform(2, 30, samples) 
    thermal_k = np.random.uniform(0.1, 0.5, samples) 
    hours = np.random.uniform(0, 87600, samples)     
    loss = np.random.uniform(-40, -20, samples)
    temp_amb = np.random.uniform(25, 85, samples)
    
    # 2. Physics
    pwr_dynamic = bias * 1.5 
    r_theta = 0.42 * (1.0 / (thermal_k / 0.3)) * np.sqrt(10000 / area)
    tj = temp_amb + (pwr_dynamic * r_theta)
    
    # 3. Baseline
    base_height = (bias * 4.0) - (np.abs(loss) * 0.5)
    base_width = 0.6 - (np.abs(loss) / 200.0)
    
    # 4. Aging
    eol_height, eol_width = apply_aging_degradation(base_height, base_width, hours, tj)
    
    # 5. Lifetime
    lifetime_years = 15.0 * (thermal_k / 0.3) * np.exp(500 / (tj + 273)) / (bias/10 + 1)
    lifetime_years = np.clip(lifetime_years, 0, 20)

    # 6. Construct Dataframe (PHYSICAL UNITS)
    df = pd.DataFrame({
        "ffe_m1": np.random.uniform(-0.12, 0, samples),
        "ffe_p1": np.random.uniform(-0.05, 0, samples),
        "area_um2": area,
        "bias_current_ma": bias,
        "rx_impedance_ohm": np.random.uniform(85, 110, samples), # Added back for schema completeness
        "thermal_k": thermal_k,
        "operating_hours": hours,
        "loss_db": loss,
        "temp_amb": temp_amb,
        # Targets
        "eye_width_ui": eol_width,
        "eye_height_mv": eol_height,
        "total_pwr_mw": pwr_dynamic,
        "tj_c": tj,
        "lifetime_years": lifetime_years
    })

    print("📏 Normalizing for JEPA Training...")
    df_scaled = scale_data(df)
    
    os.makedirs("data", exist_ok=True)
    table = pa.Table.from_pandas(df_scaled)
    pq.write_table(table, "data/samples_normalized.parquet")
    print(f"✅ Data ready at data/samples_normalized.parquet")

if __name__ == "__main__":
    generate_normalized_samples()