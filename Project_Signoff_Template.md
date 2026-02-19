# 128G SerDes Architectural Sign-off: Spatial Audit

**Project ID:** [Project Name]
**Date:** [YYYY-MM-DD]
**Status:** [PASS / FAIL / CONDITIONAL]

## I. Spatial Thermal Verdict
*Analysis of the optimized layout using the 3D Physics-NeMo twin.*

| Metric | Target | Result (Optimized) | Status |
| :--- | :--- | :--- | :--- |
| **Peak Die Temp** | < 105.0 °C | [Peak T] °C | [Pass/Fail] |
| **TX-RX Isolation** | > 300.0 um | [Dist] um | [Pass/Fail] |
| **EOL Margin (10y)** | > 0.10 UI | [Margin] UI | [Pass/Fail] |

## II. 3D Stackup Audit
*Efficiency of the 5-layer thermal conduction path.*

- **BEOL Effectiveness:** [K_eff] W/mK
- **Package Heat Spreading:** [K_eff] W/mK
- **Hotspot ROI Zoom:** Verified in `plots/user_design_thermal.png`.

## III. Multi-Physics Performance
- **Thermal Jitter Tax:** [Value] UI loss at Year 10.
- **EMI Crosstalk Penalty:** [Value] UI loss based on block proximity.
- **PPA Verdict:** [Value] pJ/bit energy efficiency at EOL.

---
## IV. Verification Checklist
1. **[ ] Voxel Integrity:** Does the power density map match the physical netlist?
2. **[ ] Corner Resilience:** Verified against FF/SS and Voltage stress?
3. **[ ] Isolation Goal:** Are sensitive analog blocks separated from DSP aggressors?

**Notes:** Sign-off is valid for the specific .itf process and package stack defined in the design JSON.
