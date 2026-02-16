# 128G SerDes Architectural Sign-off Report
**Project ID:** SERDES_128G_3NM_SIGN_OFF
**Date:** February 15, 2026
**Status:** PASS (CONDITIONAL)

## I. Physical Distribution Audit
*Analysis of heat generation sources at the Optimized Golden Configuration.*

- **Interconnect (Metal) Dissipation:** 14.2 mW 
  - *Source:* Calculated from ITF sheet resistance and 64GBaud current density.
- **Active Switching (Poly) Dissipation:** 32.8 mW 
  - *Source:* Liberty Dynamic Tables (52% Activity Factor).
- **Static Leakage (Device):** 6.4 mW 
  - *Source:* Base leakage at 25°C scaled to Tj.
- **Total Power:** 53.4 mW

## II. The Thermal-Jitter Verdict
- **Calculated Tj:** 47.4 °C (Ambient + Thermal Delta)
- **Horizontal Margin Tax:** -0.022 UI 
  - *Formula:* (47.4°C - 25°C) × 0.001 UI/°C
- **Final Horizontal Eye:** 0.498 UI (Spec: > 0.48 UI) — **PASS**

## III. PPA Performance Summary
- **Vertical Margin:** 38.5 mV (Spec: > 36.0 mV) — **PASS**
- **Energy Efficiency:** 0.417 pJ/bit (Target: < 0.60 pJ/bit) — **OPTIMAL**
- **Optimized TX-FFE Taps:** [-0.05, 0.82, -0.12, -0.01]

---
## IV. Final Verification Checklist
1. **The Linearity Test:** Verified in `plots/thermal_sensitivity.png`. Decay is linear (0.01 UI / 10°C).
2. **The DFE Guardrail Test:** DFE Tap-1 prediction within 35mV hard limits.
3. **The Cross-Check:** AI prediction matches Golden Physics within ±2%.

**Notes:** Status is CONDITIONAL pending final SI-simulation verification of Package-Die escape.
