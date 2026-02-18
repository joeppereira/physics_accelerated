import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from src.surrogate import PhysicsNeMoFNO2D
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output
from src.physics_engine import generate_spatial_layout

def analyze_package_tradeoff():
    """
    Analyzes Silicon Area vs Packaging Quality using the 2D Spatial Model.
    """
    print("📦 Analyzing Package Trade-off (2D Spatial)...")
    
    model = PhysicsNeMoFNO2D()
    try:
        model.load_state_dict(torch.load("models/spatial_fno_v1.pth", map_location=torch.device('cpu')))
        model.eval()
    except:
        return

    # Sweep K_pkg (Quality of Heat Sink)
    # We use the analytic physics solver logic for the material sweep part 
    # since the FNO wasn't trained with K_pkg as an input channel.
    # (Reusing logic from analyze_cooling_methods.py)
    
    from src.analyze_cooling_methods import analyze_cooling_tradeoff
    analyze_cooling_tradeoff()

if __name__ == "__main__":
    analyze_package_tradeoff()
