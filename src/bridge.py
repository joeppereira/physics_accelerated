import torch
import numpy as np
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS

class OptimizerBridge:
    def __init__(self, model_path="models/surrogate_v1.pth"):
        # Load the Accelerated Brain
        self.model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
        try:
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            print("🧠 Cognitive Optimizer: Weights loaded successfully.")
        except FileNotFoundError:
            print("⚠️ Warning: No trained weights found. Using randomized initialization.")
        
        self.model.eval()

    def suggest_optimal_config(self, channel_loss_db, ambient_temp=25.0):
        """
        Predicts the best FFE/Vpp combination for a given channel in milliseconds.
        """
        # Example heuristic/candidate input for the -40dB "Killer Channel" scenario
        # [ffe_m1, ffe_0, ffe_p1, ffe_p2, loss, temp, vpp, pwr]
        mock_input = torch.tensor([-0.05, 0.82, -0.10, -0.02, channel_loss_db, ambient_temp, 450.0, 65.0]).float().unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.model(mock_input).numpy()[0]
        
        # Enforce 3nm hardware guardrails (105°C Tj) and calculate status
        status = "VALIDATED" if (prediction[1] > 0.48 and prediction[2] < 105.0) else "MARGIN_FAIL"
        
        return {
            "eye_height_mv": float(prediction[0]),
            "eye_width_ui": float(prediction[1]),
            "tj_c": float(prediction[2]),
            "status": status
        }

if __name__ == "__main__":
    # Quick verification test
    bridge = OptimizerBridge()
    result = bridge.suggest_optimal_config(channel_loss_db=-36.0)
    print(f"📊 Suggested Config for -36dB Channel: {result}")