# 128G SerDes "Physics-NeMo" Digital Twin

This platform implements a **Local Physics-Informed Digital Twin** for 128G SerDes architecture. It uses **NVIDIA Modulus-style** Neural Operators (FNO) and Finite Difference Solvers to predict full 3D thermal/reliability maps in milliseconds.

## 🌟 Why Use This?

| Feature | Legacy Workflow (SPICE/FEM) | Physics-NeMo Digital Twin | **User Benefit** |
| :--- | :--- | :--- | :--- |
| **Speed** | Hours per simulation | **Milliseconds** per inference | Iterate 10,000x faster. |
| **Physics** | Lumped (Avg Temp) | **Spatial (3D Voxel)** | Detect local hotspots that kill reliability. |
| **Stackup** | Fixed / Hardcoded | **N-Layer Parametric** | Optimize metal stacks and packaging materials. |
| **Aging** | Static Guardband | **Dynamic Reliability** | Predict Year-10 failure probability today. |

---

## 🚀 Quick Start: Evaluate Your Design

You don't need to be an AI expert. Just describe your chip layout in a JSON file.

### 1. Define your Layout (`my_chip.json`)
```json
{
  "design_name": "My_Custom_SoC",
  "die_width_um": 2000,
  "die_height_um": 2000,
  "blocks": [
    {"name": "DSP", "x": 200, "y": 200, "w": 1600, "h": 600, "power_mw": 800},
    {"name": "TX",  "x": 200, "y": 1000, "w": 200,  "h": 200, "power_mw": 150},
    {"name": "RX",  "x": 600, "y": 1000, "w": 200,  "h": 200, "power_mw": 80}
  ],
  "stackup": [
    {"name": "Silicon", "type": "die", "thickness": 50, "k": 150},
    {"name": "M1", "type": "metal", "thickness": 0.1, "k": 200},
    {"name": "Substrate", "type": "pkg", "thickness": 500, "k": 20}
  ]
}
```

### 2. Run the Thermal Audit
```bash
python3 src/evaluate_design.py my_chip.json
```
**Output:**
*   Peak Temperature Report.
*   Stackup Efficiency Analysis.
*   Heatmap Plot: `plots/user_design_thermal.png`

### 3. Zoom In (Adaptive Geometry)
Need to see the gradients inside the TX block? Use the ROI flag:
```bash
# Zoom into x=0..800, y=800..1400
python3 src/evaluate_design.py my_chip.json --roi 0,800,800,1400
```

---

## 🏗 System Architecture

### 1. The Physics Engine (Teacher)
*   **Module:** `src/physics_engine.py`
*   **Method:** Finite Difference Method (FDM) solving the **3D Heat Diffusion Equation**.
*   **Capabilities:** 5-Layer Canonical Stack (Die, Metal, C4, Pkg, Board) with N-Layer Collapsing.

### 2. The AI Surrogate (Student)
*   **Module:** `src/surrogate.py`
*   **Model:** `PhysicsNeMoFNO2D` (Fourier Neural Operator).
*   **Role:** Learns the mapping from `Power_Grid` $	o$ `Temperature_Grid`.

### 3. The Optimizer (Engineer)
*   **Module:** `src/gepa.py`
*   **Algorithm:** Spatial Evolutionary Search.
*   **Goal:** Find the optimal $(x, y)$ placement to minimize thermal crosstalk.

---

## 📊 Advanced Analysis

*   **Silicon vs. Packaging:** Does a copper lid save silicon area?
    ```bash
    python3 src/analyze_package_tradeoff.py
    ```
*   **Aging vs. Spreading:** How much area is needed for 10-year life?
    ```bash
    python3 src/analyze_spatial_aging.py
    ```
