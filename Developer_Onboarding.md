# Developer Onboarding Guide

## 1. The Concept: Sensor vs. Brain
This project uses a dual-repository architecture to simulate 128G PAM4 physics.
- **Physics Core (Repo A) = The "Sensor":** It simulates the physical world (Junction Temp, Voltage Droop, Channel Loss). It provides the "Ground Truth."
- **Cognitive Optimizer (Repo B) = The "Brain":** It learns from the sensor data using a Physics-Informed Neural Network (PINN). It predicts optimal configurations without running expensive SPICE simulations.

## 2. The Thermal Loop: The "Jitter Tax"
Developers must understand the **0.01 UI per 10°C** penalty.
- **Why?** As power increases, the chip heats up.
- **Physics:** Heat increases random jitter (RJ) and degrades the eye width.
- **Rule:** `Eye_Penalty_UI = (Current_Tj - 25.0) * 0.001`
- **Result:** If your DFE burns too much power, your eye width will collapse, even if the equalization is perfect.

## 3. API Reference: OptimizerBridge
The `OptimizerBridge` connects the Physics Core to the AI model.

```python
from src.bridge import OptimizerBridge

# Initialize the Bridge
bridge = OptimizerBridge(model_path="models/surrogate_v1.pth")

# Predict Performance
# inputs: [ffe_m1, ffe_0, ffe_p1, ffe_p2, loss_db, temp_c, vpp, pwr]
prediction = bridge.predict(inputs)
# Returns: [eye_height_mv, eye_width_ui, tj_c]
```

## 4. Environment Reconstruction
To replicate the 128G SerDes platform from this lean archive:

1. **Initialize Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Proprietary Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **The Orchestration:**
   Run the master script to trigger the Physics-to-Cognitive handshake:
   ```bash
   ./run_full_cycle.sh
   ```

## 5. Core Data Structures for Developers
- **Physics Core:** All units are SI (Volts, Watts, Seconds) or industry-standard (UI, dB).
- **Cognitive Optimizer:** Normalized tensors are handled in `src/bridge.py`.
- **Mocked Data:** If you don't have access to real foundry files, `src/dummy_gen.py` creates "Synthetically Grounded" data that obeys the 3nm thermal rules.

## 6. Verification for Developers: "The Sanity Check"
To verify the environment is correctly linked:
```bash
python3 -c "import torch; from cognitive_optimizer.src.surrogate import MiniSAUFNOJEPA; print('🚀 Environment Linked Successfully')"
```
If this returns successfully, the Cognitive Optimizer is ready to consume the Physics Core data.