import torch
import numpy as np
from src.surrogate import PhysicsNeMoFNO2D

class OptimizerBridge:
    def __init__(self, model_path="models/spatial_fno_v1.pth"):
        # Load the Spatial 2D Model
        self.model = PhysicsNeMoFNO2D()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("🧠 Spatial Optimizer: 2D FNO Weights loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Warning: No trained spatial weights found.")
        
        self.model.eval()

    def predict_heatmap(self, power_grid):
        """
        Predicts the 2D Temperature Field for a given Power Layout.
        Input: (16, 16) numpy array.
        Output: (16, 16) numpy array (Temp in C).
        """
        # Normalize Input (Power / 5.0 as per training)
        scaled_input = torch.tensor(power_grid / 5.0).float().unsqueeze(0).unsqueeze(-1)
        # shape: (1, 16, 16, 1)
        
        with torch.no_grad():
            pred = self.model(scaled_input).numpy()[0, :, :, 0]
            
        # Unscale Output (Temp * 125.0)
        temp_map = pred * 125.0
        return temp_map

if __name__ == "__main__":
    bridge = OptimizerBridge()
    mock_grid = np.zeros((16, 16))
    mock_grid[8, 8] = 50.0 # Hotspot center
    temp_map = bridge.predict_heatmap(mock_grid)
    print(f"🔥 Peak Temp: {temp_map.max():.1f}C")
