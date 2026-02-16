"""
Technical Data Specs - 128G SerDes Platform

This schema defines the columns for the "Physics-Informed Dataframe" (samples_50k.parquet).
It serves as the contract between the Physics Core (Data Generator) and the Cognitive Optimizer (AI Model).

Column Definitions:
- ffe_m1, ffe_0, ffe_p1, ffe_p2: TX FFE Tap Weights (Normalized -1.0 to 1.0).
- loss_db: Channel Insertion Loss at Nyquist (dB). Positive values indicate attenuation.
- temp_c: Ambient Temperature (°C). Baseline is 25.0°C.
- vpp: Peak-to-Peak Voltage (Volts). Standard is 1.2V.
- pwr: Total Power Consumption (mW). Includes leakage + switching.

Targets:
- eye_height_mv: Vertical Eye Opening (mV). Must be > 35.0mV.
- eye_width_ui: Horizontal Eye Opening (UI). Must be > 0.48 UI.
- tj_c: Junction Temperature (°C). Includes Self-Heating Delta.

Unit Enforcement:
- All Temperatures are in Celsius (°C).
- All Margins are in Unit Intervals (UI) or Millivolts (mV).
- All Power values are in Milliwatts (mW).
"""

FEATURES = ["ffe_m1", "ffe_0", "ffe_p1", "ffe_p2", "loss_db", "temp_c", "vpp", "pwr"]
TARGETS = ["eye_height_mv", "eye_width_ui", "tj_c"]