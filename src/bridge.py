import torch
import numpy as np
from src.surrogate import PhysicsNeMoFNO2D

class OptimizerBridge:
    def __init__(self, model_path="models/spatial_fno_v1.pth"):
        self.model = PhysicsNeMoFNO2D()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("🧠 Spatial Optimizer: 3D FNO Weights loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Warning: No trained spatial weights found.")
        
        self.model.eval()

    def predict_thermal_volume(self, power_grid_layer0):
        """
        Predicts the 3D Temperature Volume.
        Input: (16, 16) Die Power Map.
        Output: (5, 16, 16) Temp Volume (Layers: Die, Metal, C4, Pkg, Board).
        """
        # Construct 5-layer input (Power only on Layer 0)
        input_vol = np.zeros((5, 16, 16))
        input_vol[0] = power_grid_layer0
        
        # Normalize
        scaled_input = torch.tensor(input_vol / 50.0).float().unsqueeze(0)
        # shape: (1, 5, 16, 16)
        
        with torch.no_grad():
            pred = self.model(scaled_input).numpy()[0]
            
        # Unscale Output (Temp * 125.0)
        temp_vol = pred * 125.0
        return temp_vol

if __name__ == "__main__":
    bridge = OptimizerBridge()
    mock_grid = np.zeros((16, 16))
    mock_grid[8, 8] = 100.0 
    temp_vol = bridge.predict_thermal_volume(mock_grid)
    print(f"🔥 Peak Die Temp: {temp_vol[0].max():.1f}C")
    print(f"❄️ Board Temp:    {temp_vol[4].max():.1f}C")