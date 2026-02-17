"""
Technical Data Specs - 128G SerDes Platform (Entropy & Aging)

This schema defines the contract between the Physics Core and the Cognitive Optimizer.
It now includes Material Physics (Thermal K) and Time-dependent Aging (Hours).
"""

FEATURES = [
    "ffe_m1", "ffe_p1", 
    "area_um2", "bias_current_ma", 
    "thermal_k",        # Effective Thermal Conductivity
    "operating_hours",  # Time in operation (0 to 87600)
    "loss_db", "temp_amb"
]

TARGETS = ["eye_width_ui", "eye_height_mv", "total_pwr_mw", "tj_c", "lifetime_years"]

NORM_FACTORS = {
    "area_um2": 15000.0,
    "bias_current_ma": 60.0,
    "thermal_k": 0.5,
    "operating_hours": 87600.0,
    "rx_impedance_ohm": 120.0,
    "eye_height_mv": 100.0,
    "total_pwr_mw": 100.0,
    "tj_c": 125.0,
    "loss_db": -40.0,
    "temp_amb": 100.0,
    "lifetime_years": 20.0
}

def scale_data(df):
    df_scaled = df.copy()
    for col, factor in NORM_FACTORS.items():
        if col in df_scaled.columns:
            df_scaled[col] = df_scaled[col] / factor
    return df_scaled

def unscale_output(preds):
    # preds: [width, height, pwr, tj, lifetime]
    width = preds[0]
    height = preds[1] * NORM_FACTORS["eye_height_mv"]
    pwr = preds[2] * NORM_FACTORS["total_pwr_mw"]
    tj = preds[3] * NORM_FACTORS["tj_c"]
    life = preds[4] * NORM_FACTORS["lifetime_years"] if len(preds) > 4 else 0.0
    return width, height, pwr, tj, life