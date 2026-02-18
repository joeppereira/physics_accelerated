import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from src.schema import NORM_FACTORS, scale_data

def solve_thermal_network(p_tx, p_rx, p_dsp, 
                          a_tx, a_rx, a_dsp, 
                          dist, k_sub, k_pkg, t_amb):
    """
    Solves a 3-Node Thermal Resistor Network.
    Nodes: TX, RX, DSP.
    Coupling: TX-RX (Distance), DSP-RX (Assumed close/layout dependent).
    """
    # 1. Self-Resistance (Vertical to Ambient)
    # R_theta = Thickness / (Area * K_pkg). 
    # Simplified: R = Const / (Area * K_pkg)
    r_tx_amb = 1000.0 / (a_tx * k_pkg + 1e-9)
    r_rx_amb = 1000.0 / (a_rx * k_pkg + 1e-9)
    r_dsp_amb = 1000.0 / (a_dsp * k_pkg + 1e-9)
    
    # 2. Mutual-Resistance (Lateral through Substrate)
    # R_mut = Distance / (CrossSection * K_sub)
    # We model heat spread.
    # Coupling TX->RX
    r_tx_rx = dist / (100.0 * k_sub + 1e-9) # 100 is effective thickness
    # Coupling DSP->RX (DSP is usually the big aggressor)
    # Assume DSP is "fixed" distance or adjacent. Let's model it as fixed dist for now.
    r_dsp_rx = 200.0 / (100.0 * k_sub + 1e-9) 

    # 3. Superposition Approximation (First Order)
    # T_node = T_amb + P_self*R_self + P_neighbor * Coupling_Factor
    # Coupling Factor ~ R_self / (R_self + R_mut) ? 
    # Simplified heat spreader logic:
    
    # DSP is the dominant heat source.
    # Delta T due to DSP power at the DSP node
    dt_dsp_self = p_dsp * r_dsp_amb
    t_dsp = t_amb + dt_dsp_self
    
    # How much of DSP heat reaches RX?
    # Thermal Divider: R_rx_amb vs R_dsp_rx
    # coupling_ratio = R_rx_amb / (R_rx_amb + R_dsp_rx)
    coupling_dsp_rx = r_rx_amb / (r_rx_amb + r_dsp_rx)
    dt_from_dsp = dt_dsp_self * coupling_dsp_rx
    
    # TX heat reaching RX
    dt_tx_self = p_tx * r_tx_amb
    coupling_tx_rx = r_rx_amb / (r_rx_amb + r_tx_rx)
    dt_from_tx = dt_tx_self * coupling_tx_rx
    
    # RX Self Heating
    dt_rx_self = p_rx * r_rx_amb
    
    # Final RX Temp (Superposition)
    t_rx = t_amb + dt_rx_self + dt_from_dsp + dt_from_tx
    
    return t_rx, t_dsp

def calc_aging(temp_c, bias_ma, area_um2, hours):
    # Arrhenius NBTI
    aging_factor = (hours / 87600.0) * np.exp(-1000.0 / (temp_c + 273.15))
    
    # EM Limit (Black's Law)
    j_density = bias_ma / (np.sqrt(area_um2) + 1e-9)
    mttf = 20.0 * (1.0 / (j_density + 0.1)) * np.exp(300.0 / (temp_c + 273.15))
    
    # Combined metric
    degradation = 0.2 * aging_factor # % loss
    return mttf, degradation

def generate_normalized_samples(samples=50000):
    print(f"🏭 Generating {samples} Spatial Multi-Physics samples...")
    
    # 1. Inputs
    a_tx = np.random.uniform(1000, 5000, samples)
    a_rx = np.random.uniform(1000, 5000, samples)
    a_dsp = np.random.uniform(5000, 20000, samples)
    dist = np.random.uniform(10, 500, samples) # TX-RX Spacing
    
    b_tx = np.random.uniform(10, 40, samples)
    b_rx = np.random.uniform(5, 20, samples)
    f_dsp = np.random.uniform(1.0, 4.0, samples) # GHz
    
    k_sub = np.random.uniform(50, 150, samples)
    k_pkg = np.random.uniform(1, 10, samples)
    
    hours = np.random.uniform(0, 87600, samples)
    loss = np.random.uniform(-40, -20, samples)
    t_amb = np.random.uniform(25, 85, samples)
    
    # 2. Power Calculation
    p_tx = b_tx * 1.2 # mW
    p_rx = b_rx * 1.0 # mW
    p_dsp = f_dsp * a_dsp * 0.005 # Scaling factor for digital dynamic power
    total_pwr = p_tx + p_rx + p_dsp
    
    # 3. Thermal Solver (The Matrix)
    # Iterate through numpy arrays? No, simpler to vectorize the math.
    # The functions above are scalar logic, let's vectorize them inline or wrap.
    # Vectorized implementation:
    
    # Resistances
    r_tx_amb = 1000.0 / (a_tx * k_pkg)
    r_rx_amb = 1000.0 / (a_rx * k_pkg)
    r_dsp_amb = 1000.0 / (a_dsp * k_pkg)
    r_dsp_rx = 200.0 / (100.0 * k_sub)
    r_tx_rx = dist / (100.0 * k_sub)
    
    # Temperatures
    t_dsp = t_amb + (p_dsp * r_dsp_amb)
    
    # RX sees heat from Self + DSP + TX
    dt_rx_self = p_rx * r_rx_amb
    dt_from_dsp = (p_dsp * r_dsp_amb) * (r_rx_amb / (r_rx_amb + r_dsp_rx))
    dt_from_tx = (p_tx * r_tx_amb) * (r_rx_amb / (r_rx_amb + r_tx_rx))
    
    t_rx = t_amb + dt_rx_self + dt_from_dsp + dt_from_tx
    
    # 4. Aging & Performance
    # Aging depends on T_rx (for analog degradation)
    # Arrhenius
    aging_factor = (hours / 87600.0) * np.exp(-1000.0 / (t_rx + 273.15))
    
    # Eye Margin
    # Base: Gain from Bias_rx + Swing from Bias_tx - Loss
    # Penalty: T_rx noise + Aging
    base_eye = (b_tx * 0.01) + (b_rx * 0.02) - (np.abs(loss) * 0.015)
    thermal_noise = (t_rx - 25.0) * 0.001
    aging_loss = 0.1 * aging_factor
    
    eye_margin = base_eye - thermal_noise - aging_loss
    eye_margin = np.clip(eye_margin, 0, 1.0)
    
    # Lifetime (Limited by hottest block, likely DSP or RX)
    # Simplified lifetime check on RX
    j_rx = b_rx / np.sqrt(a_rx)
    life_rx = 20.0 / (j_rx * np.exp(0.01 * t_rx) + 0.1)
    
    # 5. DataFrame
    df = pd.DataFrame({
        "area_tx_um2": a_tx, "area_rx_um2": a_rx, "area_dsp_um2": a_dsp,
        "dist_tx_rx_um": dist,
        "bias_tx_ma": b_tx, "bias_rx_ma": b_rx, "dsp_freq_ghz": f_dsp,
        "thermal_k_sub": k_sub, "thermal_k_pkg": k_pkg,
        "operating_hours": hours, "loss_db": loss, "temp_amb": t_amb,
        # Targets
        "eye_margin_ui": eye_margin,
        "total_pwr_mw": total_pwr,
        "tj_rx_c": t_rx,
        "tj_dsp_c": t_dsp,
        "lifetime_years": life_rx
    })

    print("📏 Normalizing...")
    df_scaled = scale_data(df)
    
    os.makedirs("data", exist_ok=True)
    table = pa.Table.from_pandas(df_scaled)
    pq.write_table(table, "data/samples_normalized.parquet")
    print(f"✅ Spatial Data Ready. RX Temp Mean: {t_rx.mean():.1f}C, DSP Temp Mean: {t_dsp.mean():.1f}C")

if __name__ == "__main__":
    generate_normalized_samples()
