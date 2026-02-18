# 128G SerDes "Physics-NeMo" Digital Twin

This platform implements a **Local Physics-Informed Digital Twin** for 128G SerDes architecture. It uses **NVIDIA Modulus-style** Neural Operators (FNO) to predict full 2D thermal/reliability maps in milliseconds, enabling spatial optimization that SPICE cannot handle.

## 🏗 System Architecture

### 1. The Physics Engine (Teacher)
*   **Module:** `src/physics_engine.py`
*   **Method:** Finite Difference Method (FDM) solving the **2D Heat Diffusion Equation**.
*   **Capabilities:** Multi-block layouts (TX, RX, DSP), thermal coupling, and package boundary conditions.

### 2. The AI Surrogate (Student)
*   **Module:** `src/surrogate.py`
*   **Model:** `PhysicsNeMoFNO2D` (2D Fourier Neural Operator).
*   **Role:** Learns the mapping from `Power_Grid (16x16)` $\to$ `Temperature_Grid (16x16)`.
*   **Speed:** $10^4\times$ faster than the FDM solver.

### 3. The Optimizer (Engineer)
*   **Module:** `src/gepa.py`
*   **Algorithm:** Spatial Evolutionary Search.
*   **Goal:** Find the optimal $(x, y)$ placement and Silicon Area to survive 10 years.

---

## 🚀 Usage Guide

### 1. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. The "Digital Twin" Cycle
Run the full pipeline to generate data, train the twin, and optimize the chip:
```bash
# 1. Generate Physics Data (Solves PDEs)
python3 src/dummy_gen_normalized.py

# 2. Train the Neural Operator
python3 src/train.py --epochs 50

# 3. Run Spatial Optimization
python3 src/gepa.py
```

### 3. Advanced Analysis
Perform specific trade-off studies:

*   **Silicon vs. Packaging:** Does a copper lid save silicon area?
    ```bash
    python3 src/analyze_package_tradeoff.py
    ```
*   **Aging vs. Spreading:** How much area is needed for 10-year life?
    ```bash
    python3 src/analyze_spatial_aging.py
    ```

---

## 📊 Key Insights
*   **Thermal Crosstalk:** Placing Digital (DSP) blocks too close ($<200\mu m$) to Analog (RX) degrades Margin by >0.05 UI due to thermal coupling.
*   **The Reliability Tax:** To survive 10 years (NBTI), silicon area must increase by ~40% vs. Day-0 requirements to lower $T_j$ below $65^{\circ}C$.