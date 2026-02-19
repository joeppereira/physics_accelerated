# 128G SerDes "Physics-NeMo" Digital Twin

This platform implements a **Local Physics-Informed Digital Twin** for 128G SerDes architecture. It uses **NVIDIA Modulus-style** Neural Operators (FNO) and Finite Difference Solvers to predict full 3D thermal/reliability maps in milliseconds.

## 🌟 Why Use This?

| Feature | Legacy Workflow (SPICE/FEM) | Physics-NeMo Digital Twin | **User Benefit** |
| :--- | :--- | :--- | :--- |
| **Speed** | Hours per simulation | **Milliseconds** per inference | Iterate 10,000x faster. |
| **Physics** | Lumped (Avg Temp) | **3D Voxel Stack** | Detect local hotspots in the BEOL. |
| **Stackup** | Fixed / Hardcoded | **N-Layer Collapsing** | Automatically merges 10+ metals into a 5-layer model. |

---

## 🚀 Design Entry & Evaluation

### 1. Hierarchical Design Format (`my_chip.json`)
The system supports nested hierarchies, allowing you to define sub-blocks within macros. Power is correctly rasterized across the voxel grid based on local density.

```json
{
  "design_name": "SerDes_v3",
  "tech_file": "config/foundry_3nm.itf",
  "blocks": [
    {
      "name": "Analog_Front_End", "x": 100, "y": 800, "w": 500, "h": 400,
      "sub_blocks": [
        {"name": "TX_Drv", "x": 10, "y": 10, "w": 100, "h": 100, "power_mw": 120},
        {"name": "RX_LNA", "x": 200, "y": 10, "w": 100, "h": 100, "power_mw": 40}
      ]
    }
  ]
}
```

### 2. Adaptive ROI Zoom
Standard resolution is $16 \times 16$ across the die. To analyze a specific dense region (like the RX input stage) at higher fidelity, use the **Region of Interest (ROI)** flag. This "zooms" the physics grid into the specified window.

```bash
# Evaluate full die
python3 src/evaluate_design.py my_chip.json

# Zoom into the AFE region (x:100-600, y:800-1200) for high-res thermal audit
python3 src/evaluate_design.py my_chip.json --roi 100,800,600,1200
```

### 3. Thermal Stackup Specification
The `tech_file` (.itf) defines the metal layers. The system automatically collapses these into 5 canonical layers:
1. **Die:** Active silicon heating.
2. **BEOL:** Homogenized metal stack (M1-M10).
3. **Interconnect:** C4 Bumps/Underfill layer.
4. **Package:** Substrate/Core material.
5. **Board:** PCB and Heat Sink interface.

---

## 🏗 System Architecture

### 1. The Physics Factory (`serdes_architect`)
High-fidelity ground truth generator. Run `python3 src/data_gen.py` in the sibling repo to refresh training data.

### 2. The AI Surrogate (`physics_accelerated`)
5-channel Fourier Neural Operator. Learns the spatial PDE solution to predict temperature fields instantly.

### 3. Advanced Analysis Suite
*   **Aging Analysis:** `src/analyze_spatial_aging.py` - Tracks Eye Margin vs Area over 10 years.
*   **Isolation Audit:** `src/analyze_isolation.py` - Quantifies Crosstalk vs Distance.
*   **PVT Corners:** `src/analyze_pvt_corners.py` - Tests FF/SS and Voltage Stress scenarios.

---

## 📋 Documentation Reference
- **[Results.md](Results.md):** Detailed comparison between scalar and spatial modeling.
- **[reports/signoff_report.md](reports/signoff_report.md):** The latest architectural verdict.
- **[Developer_Onboarding.md](Developer_Onboarding.md):** Environment setup guide.
