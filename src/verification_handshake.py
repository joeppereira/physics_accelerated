import argparse
import torch
import os
from src.bridge import SCO_Predictor

def run_handshake():
    print("🤝 Starting Mutual Handshake Verification...")
    
    # 1. Verification A: Data Integrity (Simulated)
    # In a real scenario, this would read 'data/samples_50k.parquet'
    print("\n[Step 1] Verification A: Checking Data Integrity...")
    from src.model_utils import extract_tech_constraints
    
    # Create a dummy parquet for testing if it doesn't exist
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    dummy_parquet = "data/test_samples.parquet"
    if not os.path.exists("data"):
        os.makedirs("data")
        
    df = pd.DataFrame({'a': [1], 'b': [2]})
    table = pa.Table.from_pandas(df)
    meta = {
        b'technology_node': b'3nm_TSMC_N3E',
        b'dfe_tap1_limit_mv': b'35.0',
        b'thermal_ceiling_c': b'105.0'
    }
    table = table.replace_schema_metadata(meta)
    pq.write_table(table, dummy_parquet)
    
    rules, constraints = extract_tech_constraints(dummy_parquet)
    if constraints['dfe_limit'] == 35.0:
        print("✅ Data Integrity Pass: DFE Limit 35mV detected.")
    else:
        print(f"❌ Data Integrity Fail: Detected {constraints['dfe_limit']}")

    # 2. Verification C: Usage as Library (Bridge Test)
    print("\n[Step 2] Verification C: Testing Bridge Interface...")
    predictor = SCO_Predictor(model_weight_path="models/surrogate_v1.pth")
    
    test_input = {
        "ffe_c_minus_1": -0.1, 
        "ffe_c_0": 0.9, 
        "ffe_c_plus_1": -0.05, 
        "ffe_c_plus_2": -0.02,
        "channel_loss_db": -20.0, 
        "ambient_temp_c": 25.0, 
        "v_pp_mv": 800.0,
        "vga_gain": 2.0
    }
    
    result = predictor.get_cognitive_margin(test_input)
    print(f"✅ Bridge Prediction Received: {len(result['prediction_vector'])} scalars")
    
    print("\n🤝 Handshake Complete.")

if __name__ == "__main__":
    run_handshake()
