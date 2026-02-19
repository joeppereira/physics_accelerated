# Developer Onboarding Guide: 3D Physics-NeMo Edition

## 1. The Multi-Repo Architecture
This project uses a "Sensor-Brain" architecture across two repositories:
- **Physics Core (`serdes_architect`):** The ground truth generator. Uses a 3D FDM solver to produce high-fidelity thermal maps.
- **Cognitive Optimizer (`physics_accelerated`):** The AI brain. Trains 2D Fourier Neural Operators (FNO) on the 3D voxel data for millisecond-scale optimization.

## 2. Environment Reconstruction
To replicate the 128G SerDes platform:

1. **Initialize Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Sync the Physics Factory:**
   Ensure `serdes_architect` is updated and has generated the 3D tensors:
   ```bash
   cd ../serdes_architect
   python3 src/data_gen.py
   ```

3. **The AI Handshake:**
   In `physics_accelerated`, run the training cycle to ingest the new 3D data:
   ```bash
   ./run_full_cycle.sh
   ```

## 3. Core Data Structures
- **Input Tensors (`x_3d.pt`):** Shape `(Batch, 5, 16, 16)`. Channels represent the 5 canonical layers: [Die, BEOL, C4, Package, Board].
- **Output Tensors (`y_3d.pt`):** Shape `(Batch, 5, 16, 16)`. Predicted temperature maps for each layer.
- **Hierarchical JSON:** Designs are defined in nested JSON formats, supporting sub-blocks and custom ROI (Region of Interest) zoom.

## 4. Verification: "The Sanity Check"
Verify the 3D AI brain is loaded correctly:
```bash
python3 -c "import torch; from src.surrogate import PhysicsNeMoFNO2D; print('🚀 3D Spatial FNO Linked successfully')"
```
