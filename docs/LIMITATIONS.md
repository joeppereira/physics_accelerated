# System Limitations & Operating Boundaries

**Critical Notice:** The `physics_accelerated` platform is an **Architectural Pathfinder**. It calculates trends and trade-offs ($A$ is better than $B$) with high confidence, but absolute values ($T_j = 57.21^{\circ}C$) have specific error bars and validity limits compared to commercial Golden Tools (Ansys/Cadence).

## 1. Physics Fidelity Gaps

| Domain | Current Implementation | The "Real World" Gap | Risk |
| :--- | :--- | :--- | :--- |
| **Thermal** | **Linear Conduction:** $K$ is constant. | **Non-Linear:** $K_{Si}(T)$ drops by ~30% at $100^{\circ}C$. | Model **underestimates** peak temps at extreme heat ($>100^{\circ}C$). |
| **Electrical** | **Resistive Mesh:** DC Voltage Drop only. | **RLC Network:** Includes Inductance ($L$) and AC Switching noise. | Misses $L \cdot di/dt$ noise (voltage droop during fast switching). |
| **Geometry** | **Voxel ($64 \times 64$):** Resolution $\approx 30 \mu m$. | **GDSII:** Resolution $\approx 0.003 \mu m$. | Cannot detect "Self-Heating" of individual nanowires or vias. |
| **Cooling** | **Lumped BC:** Top surface = $G_{amb}$. | **CFD:** Complex airflow, turbulence, and radiation. | Optimistic for passive cooling; realistic for active heat sinks. |

## 2. AI Surrogate Boundaries (The "Trust Zone")

The FNO Model (`PhysicsNeMoFNO2D`) is an interpolator. It fails if asked to extrapolate far beyond its training data.

*   **Valid Power Range:** $0 - 500 mW$ per block.
*   **Valid Conductivity ($K$):** $1.0 - 400.0 W/mK$.
*   **Failure Mode:** If you input a material with $K=2000$ (Diamond), the AI will likely cap the prediction at the behavior of Copper ($K=400$), leading to massive errors.

## 3. Aging Model Limits

*   **Mechanism:** First-Order Arrhenius (NBTI) + Black's Law (EM).
*   **Missing:** Stress Migration, HCI (Hot Carrier Injection) dependence on specific waveform toggling, and Mechanical Fatigue (thermal cycling cracks).
*   **Result:** The "Life-Cycle" prediction is a **Best Case** scenario. Real-world mechanical failures often happen before electrical wear-out.

## 4. Usage Recommendation

*   **DO USE FOR:** Floorplanning, Material Selection, Packaging Trade-offs, Pre-Silicon prototyping.
*   **DO NOT USE FOR:** Final Tape-out Sign-off, Safety-Critical Failure Analysis, Precise Yield Estimation.
