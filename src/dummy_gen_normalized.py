import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from src.schema import NORM_FACTORS, scale_data

def apply_aging_degradation(eye_height, eye_width, hours, temp_c):
    """
    Simulates the 10-year decay of 3nm transistors.
    Higher Temp + Higher Hours = Faster Margin Collapse.
    """
    # Aging factor: accelerated by Arrhenius equation (Temperature)
    # Using 273.15 for Kelvin conversion
    aging_factor = (hours / 87600.0) * np.exp(-1000.0 / (temp_c + 273.15 + 1e-9))
    
    # 10-year EOL (End of Life) margins
    eol_height = eye_height * (1.0 - (0.15 * aging_factor)) # -15% Height
    eol_width = eye_width - (0.05 * aging_factor)          # +0.05 UI Jitter
    
    return eol_height, eol_width

def generate_normalized_samples(samples=50000):
    print(f"🏭 Generating {samples} samples with MATERIAL & AGING PHYSICS...")
    
    # 1. Inputs
    area = np.random.uniform(5000, 15000, samples)
    bias = np.random.uniform(2, 30, samples) 
    thermal_k = np.random.uniform(0.1, 0.5, samples) # Material Conductivity
    hours = np.random.uniform(0, 87600, samples)     # Age of the device
    loss = np.random.uniform(-40, -20, samples)
    temp_amb = np.random.uniform(25, 85, samples)
    
    # 2. Fourier Law of Conduction (Thermal Knob)
    # Delta_T = Power / (Area * k_eff)
    # We'll normalize area to um^2 and k to a relative scalar.
    pwr_dynamic = bias * 1.5 # mW proxy
    
    # Effective R_theta improves (lowers) with higher thermal_k
    # R_theta = 1.0 / (Area_norm * k_eff)
    r_theta = 0.42 * (1.0 / (thermal_k / 0.3)) * np.sqrt(10000 / area)
    tj = temp_amb + (pwr_dynamic * r_theta)
    
    # 3. Baseline Performance (Day Zero)
    base_height = (bias * 4.0) - (abs(loss) * 0.5)
    base_width = 0.6 - (abs(loss) / 200.0)
    
    # 4. Apply AGING TAX
    eol_height, eol_width = apply_aging_degradation(base_height, base_width, hours, tj)
    
    # 5. MTTF Calculation (for the lifetime_years target)
    # Simple Arrhenius-based life prediction
    lifetime_years = 15.0 * (thermal_k / 0.3) * np.exp(500 / (tj + 273)) / (bias/10 + 1)
    lifetime_years = np.clip(lifetime_years, 0, 20)

    # 6. Construct Dataframe
    df = pd.DataFrame({
        "ffe_m1": np.random.uniform(-0.12, 0, samples),
        "ffe_p1": np.random.uniform(-0.05, 0, samples),
        "area_um2": area,
        "bias_current_ma": bias,
        "thermal_k": thermal_k,
        "operating_hours": hours,
        "loss_db": loss,
        "temp_amb": temp_amb,
        # Targets (Physical Units for now)
        "eye_width_ui": eol_width,
        "eye_height_mv": eol_height,
        "total_pwr_mw": pwr_dynamic, # Simplified
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
