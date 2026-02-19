# 128G SerDes "Physics-NeMo" Digital Twin

This platform implements a **Local Physics-Informed Digital Twin** for 128G SerDes architecture. It uses **NVIDIA Modulus-style** Neural Operators (FNO) and Finite Difference Solvers to predict full 3D thermal/reliability maps in milliseconds.

## 🌟 Why Use This?

| Feature | Legacy Workflow (SPICE/FEM) | Physics-NeMo Digital Twin | **User Benefit** |
| :--- | :--- | :--- | :--- |
| **Speed** | Hours per simulation | **Milliseconds** per inference | Iterate 10,000x faster. |
| **Foundry Import**| Manual data entry | **Direct .itf Parsing** | Direct link to foundry process. |
| **Physics** | Lumped (Avg Temp) | **3D Voxel Stack** | Detect local hotspots in the BEOL. |
| **Stackup** | Fixed / Hardcoded | **N-Layer Collapsing** | Automatically merges 10+ metals into a 5-layer model. |

---

## 🚀 Quick Start: Evaluate Your Design

### 1. Link your Foundry Tech (`foundry.itf`)
The system parses standard Interconnect Technology Format (.itf) files to extract layer thicknesses and material properties.

### 2. Define your Design (`my_chip.json`)
Point to your tech file and define your block coordinates:
```json
{
  "design_name": "SerDes_v2",
  "tech_file": "config/foundry_3nm.itf",
  "blocks": [
    {"name": "DSP", "x": 200, "y": 200, "w": 1600, "h": 600, "power_mw": 800}
  ]
}
```

### 3. Run the Thermal Audit
```bash
python3 src/evaluate_design.py my_chip.json
```
**Process Flow:**
1.  **Parse:** Extracts metal layers from `.itf`.
2.  **Collapse:** Uses the **Thermal Resistance Rule** to merge N-layers into the 5-layer AI-compatible stack.
3.  **Solve:** Executes 3D FDM solver using local power densities.
4.  **Viz:** Generates `plots/user_design_thermal.png`.

---

## 🏗 System Architecture

### 1. The Physics Factory (`serdes_architect`)
*   **Role:** Acts as the high-fidelity ground truth generator.
*   **Input:** Multi-layer power volumes.
*   **Output:** `x_3d.pt` and `y_3d.pt` tensors for AI training.

### 2. The AI Surrogate (`physics_accelerated`)
*   **Model:** `PhysicsNeMoFNO2D` (5-channel Fourier Neural Operator).
*   **Inference:** Predicts the temperature at every voxel across the 5 canonical layers.

### 3. The Automation Suite
*   **Adaptive ROI:** Use `--roi xmin,ymin,xmax,ymax` to zoom into hotspots.
*   **PVT Corners:** Check `src/analyze_pvt_corners.py` for FF/SS/Voltage stress tests.

---

## 📋 Documentation Reference
- **[Results.md](Results.md):** Detailed comparison between scalar and spatial modeling.
- **[reports/signoff_report.md](reports/signoff_report.md):** The latest architectural verdict.
- **[Developer_Onboarding.md](Developer_Onboarding.md):** Guide for setting up the 2-repo environment.