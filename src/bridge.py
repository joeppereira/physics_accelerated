import torch
import numpy as np
from src.surrogate import PhysicsNeMoFNO2D

class OptimizerBridge:
    def __init__(self, model_path="models/spatial_fno_v1.pth"):
        self.model = PhysicsNeMoFNO2D()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print("🧠 Spatial Optimizer: Super-Res 64x64 FNO Weights loaded.")
        except FileNotFoundError:
            print("⚠️ Warning: No trained spatial weights found.")
        
        self.model.eval()

    def predict_thermal_volume(self, power_grid_layer0, k_layers=None):
        """
        Predicts 3D Temperature.
        Input: power_grid_layer0: (64, 64)
        """
        N = 64
        # 1. Power Volume (5, N, N)
        p_vol = np.zeros((5, N, N))
        p_vol[0] = power_grid_layer0
        
        # 2. Material Volume (5, N, N)
        k_vol = np.zeros((5, N, N))
        if k_layers:
            for l in range(5):
                k_vol[l] = k_layers[l]
        else:
            k_vol[0]=150; k_vol[1]=300; k_vol[2]=50; k_vol[3]=10; k_vol[4]=0.5
            
        # 3. Stack (10, N, N)
        x_sample = np.concatenate([p_vol / 50.0, k_vol / 400.0], axis=0)
        tensor_in = torch.tensor(x_sample).float().unsqueeze(0)
        
        with torch.no_grad():
            pred = self.model(tensor_in).numpy()[0]
            
        return pred * 125.0

if __name__ == "__main__":
    pass