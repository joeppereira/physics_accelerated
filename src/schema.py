"""
Technical Data Specs - 128G SerDes Platform (Distributed Thermal-Aging Model)

This schema defines a Multi-Block Spatial Physics model.
We now model the TX, RX, and DSP as distinct thermal nodes with mutual coupling.
"""

FEATURES = [
    # Geometry & Layout
    "area_tx_um2",   # Transmitter Analog
    "area_rx_um2",   # Receiver Analog (Sensitive)
    "area_dsp_um2",  # Digital Core (Aggressor)
    "dist_tx_rx_um", # Spacing between TX and RX
    
    # Electrical Config
    "bias_tx_ma",    # Driver strength
    "bias_rx_ma",    # LNA/VGA strength
    "dsp_freq_ghz",  # Digital switching rate
    
    # Material Physics
    "thermal_k_sub", # Substrate conductivity (Heat spread sideways)
    "thermal_k_pkg", # Package conductivity (Heat escape up/down)
    
    # Lifecycle
    "operating_hours", 
    "loss_db", 
    "temp_amb"
]

# We now track separate temperatures to see "Thermal Crosstalk"
TARGETS = [
    "eye_margin_ui",    # Final Performance at RX
    "total_pwr_mw",
    "tj_rx_c",          # RX Temp (Critical for Noise/Aging)
    "tj_dsp_c",         # DSP Temp (Critical for Global Heat)
    "lifetime_years"    # Limited by the hottest block
]

NORM_FACTORS = {
    # Areas
    "area_tx_um2": 5000.0,
    "area_rx_um2": 5000.0,
    "area_dsp_um2": 20000.0,
    "dist_tx_rx_um": 500.0, # Max spacing
    
    # Power
    "bias_tx_ma": 40.0,
    "bias_rx_ma": 20.0,
    "dsp_freq_ghz": 4.0,
    
    # Physics
    "thermal_k_sub": 150.0, # Silicon ~150 W/mK
    "thermal_k_pkg": 10.0,  # Thermal Interface Material
    
    # Env
    "operating_hours": 87600.0,
    "loss_db": -40.0,
    "temp_amb": 100.0,
    
    # Outputs
    "eye_margin_ui": 1.0,
    "total_pwr_mw": 500.0, # Digital is power hungry
    "tj_rx_c": 125.0,
    "tj_dsp_c": 125.0,
    "lifetime_years": 20.0
}

def scale_data(df):
    df_scaled = df.copy()
    for col, factor in NORM_FACTORS.items():
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col] / factor
    return df_scaled

def unscale_output(preds):
    # preds: [margin, pwr, tj_rx, tj_dsp, lifetime]
    margin = preds[0]
    pwr = preds[1] * NORM_FACTORS["total_pwr_mw"]
    tj_rx = preds[2] * NORM_FACTORS["tj_rx_c"]
    tj_dsp = preds[3] * NORM_FACTORS["tj_dsp_c"]
    life = preds[4] * NORM_FACTORS["lifetime_years"] if len(preds) > 4 else 0.0
    return margin, pwr, tj_rx, tj_dsp, life
