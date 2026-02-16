import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from src.schema import scale_data, FEATURES, TARGETS

def generate(samples=50000, output_path="data/samples_50k.parquet"):
    print(f"🏭 Generating {samples} physics-grounded samples (Multi-Knob Trade-off)...")
    
    # 1. Knob Distributions (Features)
    ffe_m1 = np.random.uniform(-0.2, 0.0, samples)
    ffe_p1 = np.random.uniform(-0.1, 0.0, samples)
    
    # Area Knob: 5000 um^2 (Dense) to 15000 um^2 (Sparse/Cool)
    area_um2 = np.random.uniform(5000, 15000, samples)
    
    # Current Knob: 10mA (Low Power/Low Speed) to 60mA (High Performance)
    bias_current_ma = np.random.uniform(10, 60, samples)
    
    # Impedance Knob: 85 Ohm to 115 Ohm (Target 100 Ohm)
    rx_impedance_ohm = np.random.uniform(85, 115, samples)
    
    # Environmental/Channel
    loss_db = np.random.uniform(-35, -10, samples)
    temp_amb = np.random.uniform(25, 85, samples)
    
    # 2. Physics Core: Architecture Rules
    
    # Rule A: Area Knob -> Thermal Resistance (R_theta)
    # R_theta_JA = 0.42 * sqrt(10000 / area_um2)
    r_theta_ja = 0.42 * np.sqrt(10000.0 / area_um2)
    
    # Rule B: Current Knob -> Active Power
    # P_active = I_bias * Vdd * Freq (Normalized Freq=1.0, Vdd=1.0V)
    p_active_mw = bias_current_ma * 1.0 
    
    # Rule C: Impedance Knob -> Reflection Loss (Signal Integrity Penalty)
    reflection_coeff = np.abs((rx_impedance_ohm - 100.0) / (rx_impedance_ohm + 100.0))
    signal_integrity_factor = 1.0 - reflection_coeff
    
    # 3. The Thermal Loop (Self-Heating)
    # Tj = Ambient + (Power * R_theta)
    # But Power depends on Temp (Leakage)!
    base_leakage_mw = 5.0 
    
    # First pass
    tj_est = temp_amb + ((p_active_mw + base_leakage_mw) * r_theta_ja)
    
    # Second pass (Refined Leakage)
    leakage_mw = base_leakage_mw * np.exp(0.015 * (tj_est - 25.0))
    total_pwr_mw = p_active_mw + leakage_mw
    
    tj_c = temp_amb + (total_pwr_mw * r_theta_ja)
    
    # 4. The Tax: Jitter Penalty
    # Rule: 0.01 UI per 10°C above 25°C -> 0.001 UI per 1°C
    thermal_tax_ui = np.maximum(0, (tj_c - 25.0) * 0.001)
    
    # 5. Targets Calculation
    # Eye Height: Proportional to Bias Current (Gain) and Signal Integrity, penalized by Loss
    gain_factor = np.clip(bias_current_ma / 30.0, 0.5, 1.5)
    ffe_dist = np.sqrt((ffe_m1 - (-0.15))**2 + (ffe_p1 - (-0.05))**2)
    ffe_penalty = ffe_dist * 20.0 # mV penalty
    
    base_height = 50.0 - (np.abs(loss_db) * 0.8) # Loss degradation
    eye_height_mv = (base_height * gain_factor * signal_integrity_factor) - ffe_penalty
    eye_height_mv = np.maximum(0, eye_height_mv)
    
    # Eye Width: Base width reduced by Thermal Tax and Noise
    base_width = 0.65 - (np.abs(loss_db) / 200.0)
    slew_bonus = (bias_current_ma - 20.0) * 0.002
    
    eye_width_ui = base_width + slew_bonus - thermal_tax_ui
    eye_width_ui = np.maximum(0, eye_width_ui)

    # 6. Assemble DataFrame
    df = pd.DataFrame({
        "ffe_m1": ffe_m1, "ffe_p1": ffe_p1,
        "area_um2": area_um2, "bias_current_ma": bias_current_ma,
        "rx_impedance_ohm": rx_impedance_ohm,
        "loss_db": loss_db, "temp_amb": temp_amb,
        "eye_width_ui": eye_width_ui,
        "eye_height_mv": eye_height_mv,
        "total_pwr_mw": total_pwr_mw,
        "tj_c": tj_c
    })
    
    # NORMALIZE BEFORE SAVING
    print("📏 Normalizing Features and Targets...")
    df_scaled = scale_data(df)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    table = pa.Table.from_pandas(df_scaled)
    pq.write_table(table, output_path)
    
    print(f"✅ Physics data created at {output_path} (Normalized)")
    print(f"   - R_theta range: {r_theta_ja.min():.3f} - {r_theta_ja.max():.3f} C/mW")
    print(f"   - Power range: {total_pwr_mw.min():.1f} - {total_pwr_mw.max():.1f} mW")
    print(f"   - Tax range: {thermal_tax_ui.min():.3f} - {thermal_tax_ui.max():.3f} UI")

if __name__ == "__main__":
    generate()