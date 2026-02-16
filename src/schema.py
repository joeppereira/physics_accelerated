"""
Technical Data Specs - 128G SerDes Platform (Normalized)

This schema defines the columns for the "Physics-Informed Dataframe" (samples_50k.parquet).
It serves as the contract between the Physics Core (Data Generator) and the Cognitive Optimizer (AI Model).

Column Definitions:
- ffe_m1, ffe_p1: TX FFE Tap Weights (Normalized -1.0 to 1.0).
- area_um2: Chip Area allocation per lane (um^2).
- bias_current_ma: RX Analog Bias Current (mA).
- rx_impedance_ohm: Receiver Termination Impedance (Ohms).
- loss_db: Channel Insertion Loss at Nyquist (dB).
- temp_amb: Ambient Temperature (°C).

Targets:
- eye_width_ui: Horizontal Eye Opening (UI).
- eye_height_mv: Vertical Eye Opening (mV).
- total_pwr_mw: Total Power Consumption (mW).
- tj_c: Junction Temperature (°C).
"""

FEATURES = [
    "ffe_m1", "ffe_p1", 
    "area_um2", "bias_current_ma", "rx_impedance_ohm", 
    "loss_db", "temp_amb"
]

TARGETS = ["eye_width_ui", "eye_height_mv", "total_pwr_mw", "tj_c"]

# Normalization Factors (Max Values approx)
NORM_FACTORS = {
    "area_um2": 15000.0,
    "bias_current_ma": 60.0, # Updated from 15.0 to 60.0 based on dummy_gen range (10-60)
    "rx_impedance_ohm": 120.0,
    "eye_height_mv": 100.0,
    "total_pwr_mw": 100.0,
    "tj_c": 125.0,
    "loss_db": -40.0,    # Normalize negative loss
    "temp_amb": 100.0
}

def scale_data(df):
    """Normalize features to a [0, 1] range for NN stability."""
    df_scaled = df.copy()
    for col, factor in NORM_FACTORS.items():
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col] / factor
    return df_scaled

def unscale_output(preds):
    """Restores scaled predictions to physical units."""
    # preds: [width, height, pwr, tj]
    # width is not scaled in NORM_FACTORS (factor=1)
    width = preds[0]
    height = preds[1] * NORM_FACTORS["eye_height_mv"]
    pwr = preds[2] * NORM_FACTORS["total_pwr_mw"]
    tj = preds[3] * NORM_FACTORS["tj_c"]
    return width, height, pwr, tj