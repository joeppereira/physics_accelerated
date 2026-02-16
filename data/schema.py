# Shared Interface Schema
SCHEMA_DEFINITION = {
    "inputs": [
        "ffe_c_minus_1", "ffe_c_0", "ffe_c_plus_1", "ffe_c_plus_2",
        "channel_loss_db", "ambient_temp_c", "v_pp_mv", "vga_gain"
    ],
    "outputs": {
        "stages": [0, 1, 2, 3, 4, 5, 6], # Vert_mv and Horiz_ui for each
        "max_junction_temp_c": float,
        "total_power_mw": float,
        "dfe_1_mv": float,
        # --- Spatial Thermal Metrics (Icepak Surrogate) ---
        "temp_stage_0_c": float,      # Input Stage (Cooler)
        "temp_stage_6_c": float,      # Output Driver (Hotter)
        "temp_diff_pair_delta_c": float # Delta T betwen P and N legs
    },
    "metadata_keys": [
        "technology_node", "dfe_tap1_limit_mv", "spec_standard"
    ]
}
