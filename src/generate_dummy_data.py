import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os

def generate_dummy_data(output_path="data/samples_50k.parquet", n_samples=100):
    print(f"🧪 Generating {n_samples} physics-correlated samples (Thermal 3D Enabled)...")
    
    # 1. Generate Inputs
    ffe_c_minus_1 = np.random.uniform(-0.15, 0.05, n_samples)
    ffe_c_0       = np.random.uniform(0.6, 1.0, n_samples)
    ffe_c_plus_1  = np.random.uniform(-0.2, 0.0, n_samples)
    ffe_c_plus_2  = np.random.uniform(-0.1, 0.0, n_samples)
    
    channel_loss_db = np.random.uniform(-40.0, -10.0, n_samples)
    ambient_temp_c  = np.random.uniform(25.0, 85.0, n_samples)
    v_pp_mv         = np.random.uniform(800.0, 1000.0, n_samples)
    vga_gain        = np.random.uniform(1.0, 8.0, n_samples)

    data = {
        "ffe_c_minus_1": ffe_c_minus_1,
        "ffe_c_0": ffe_c_0,
        "ffe_c_plus_1": ffe_c_plus_1,
        "ffe_c_plus_2": ffe_c_plus_2,
        "channel_loss_db": channel_loss_db,
        "ambient_temp_c": ambient_temp_c,
        "v_pp_mv": v_pp_mv,
        "vga_gain": vga_gain
    }
    
    # 2. Physics & Thermal Logic (Simulating Icepak)
    # -------------------------------------------------------------------------
    
    # A. Base Power Calculation
    # Driver Power (Stage 6) dominates and scales with V_pp^2
    driver_power = 150.0 + ((v_pp_mv - 800.0) / 200.0) * 100.0 
    # Analog FE Power (Stage 0-5) scales with VGA Gain
    afe_power = 50.0 + (vga_gain * 10.0)
    
    data["total_power_mw"] = driver_power + afe_power + np.random.normal(0, 5, n_samples)
    
    # B. Spatial Thermal Mapping (The "3D" element)
    # Logic: The die has a thermal resistance map. Stage 6 is the heat source.
    # Heat flows from Stage 6 -> Stage 0.
    
    theta_ja_driver = 0.12 # C/mW (Hotter spot)
    theta_ja_input  = 0.08 # C/mW (Further from heat source)
    
    # Temp Stage 6 (Driver) = Ambient + (DriverPower * Local_Theta) + (AFE_Power * Coupling_Factor)
    data["temp_stage_6_c"] = ambient_temp_c + (driver_power * theta_ja_driver) + (afe_power * 0.02)
    
    # Temp Stage 0 (Input) = Ambient + (AFE_Power * Local_Theta) + (DriverPower * Spread_Factor)
    # Notice Spread_Factor (0.04) simulates heat conducting across the silicon substrate from Stg6 to Stg0
    data["temp_stage_0_c"] = ambient_temp_c + (afe_power * theta_ja_input) + (driver_power * 0.04)
    
    # Max Junction is essentially the Driver temp
    data["max_junction_temp_c"] = data["temp_stage_6_c"]

    # C. Differential Asymmetry (P vs N Leg)
    # Logic: Layout mismatch + High Current = Delta T
    # Higher V_pp = Higher Current.
    random_mismatch_factor = np.random.normal(0.0, 1.0, n_samples) # Process variation
    current_drive_factor = v_pp_mv / 1000.0
    
    # The Delta T between P and N legs
    data["temp_diff_pair_delta_c"] = np.abs(random_mismatch_factor * current_drive_factor * 2.0)


    # 3. Signal Quality (Standard)
    # -------------------------------------------------------------------------
    base_eye = 200.0 
    loss_penalty = (channel_loss_db + 10.0) * 5.0
    # Use spatial temp! Stage 6 eye degrades based on Stage 6 temp.
    temp_penalty = (data["temp_stage_6_c"] - 25.0) * 0.6
    swing_boost = (v_pp_mv - 800.0) * 0.2
    
    main_eye = base_eye + loss_penalty - temp_penalty + swing_boost
    main_eye = np.maximum(main_eye, 0.0)

    for i in range(7):
        data[f"stage_{i}_eye_height_mv"] = main_eye * (0.5 + i/12.0) + np.random.normal(0, 5, n_samples)
        data[f"stage_{i}_eye_width_ui"] = (data[f"stage_{i}_eye_height_mv"] / 300.0) * 0.8
        
    data["dfe_1_mv"] = np.abs(channel_loss_db) * 1.2 + np.random.normal(0, 2, n_samples)

    df = pd.DataFrame(data)
    
    # 4. Save
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    table = pa.Table.from_pandas(df)
    meta = {
        b'technology_node': b'3nm',
        b'dfe_tap1_limit_mv': b'35.0',
        b'thermal_ceiling_c': b'105.0'
    }
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, output_path)
    print(f"✅ Dummy data saved to {output_path}")

if __name__ == "__main__":
    generate_dummy_data()
