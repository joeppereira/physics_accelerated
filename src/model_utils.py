import pandas as pd
import pyarrow.parquet as pq
import torch

def extract_tech_constraints(parquet_path):
    """
    Reads the 'Skills' and 'Rules' embedded in the Parquet metadata.
    Ensures Repo B respects the 3nm hardware limits of Repo A.
    """
    try:
        table = pq.read_table(parquet_path)
        meta = table.schema.metadata
        
        # Decode metadata bytes to strings if they exist, otherwise use empty dict
        if meta:
            tech_rules = {k.decode(): v.decode() for k, v in meta.items()}
        else:
            tech_rules = {}
        
        # Extract specific constraints for the Loss Function
        constraints = {
            "dfe_limit": float(tech_rules.get("dfe_tap1_limit_mv", 35.0)),
            "tj_max": float(tech_rules.get("thermal_ceiling_c", 105.0)),
            "target_eff": float(tech_rules.get("energy_efficiency_limit_pj_bit", 0.60))
        }
        
        print(f"🛠️ Technology Rules Loaded: {tech_rules.get('technology_node', 'Unknown')} Node")
        return tech_rules, constraints
    except Exception as e:
        print(f"Warning: Could not read metadata from {parquet_path}: {e}")
        # Return defaults if file not found or error
        return {}, {
            "dfe_limit": 35.0,
            "tj_max": 105.0,
            "target_eff": 0.60
        }

def prepare_tensors(df, input_cols, output_cols):
    """Converts Pandas/Parquet data to Torch tensors for the FNO."""
    X = torch.tensor(df[input_cols].values).float()
    y = torch.tensor(df[output_cols].values).float()
    return X, y
