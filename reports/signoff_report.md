# 128G SerDes Architectural Sign-off Report
**Generated on:** 2026-02-15 22:24:24

## I. Design Identity & Process
- **Target Node:** 3nm_FinFET (LVT/SVT Stack)
- **Protocol:** PCIe 7.0 / PAM4 @ 128 Gbps
- **Operating Corner:** Worst-Case (1.1x Voltage, 125°C Ambient)

## II. Physical Distribution Audit
- **Metal Interconnect Dissipation:** 9.63 mW
- **Device Switching (Poly) Dissipation:** 41.73 mW
- **Static Leakage:** 12.84 mW
- **Peak Power Density:** 535.000 W/mm² at DFE Summer

## III. The "Thermal-Jitter" Verdict
- **Steady-State Tj:** 98.5 °C
- **Calculated Thermal Tax:** 0.073 UI
- **Final Horizontal Margin:** 0.495 UI (Spec: > 0.48 UI)
- **Status:** ✅ PASSED

## IV. PPA Performance Summary
- **Vertical Margin:** 38.20 mV (Spec: > 36 mV)
- **Energy Efficiency:** 0.501 pJ/bit (Target: < 0.60 pJ/bit)
- **Optimized FFE Taps:** [-0.05, 0.82, -0.12, -0.01]
- **Target Channel Loss:** -36.0 dB

---
## V. Final Verification Checklist
1. **The Linearity Test:** Verified in `plots/thermal_sensitivity.png`. Decay is linear (0.01 UI / 10°C).
2. **The DFE Guardrail Test:** DFE Tap-1 prediction within 3nm limits.
3. **The Cross-Check:** Predicted Eye Height 38.2mV is within ±2% of Golden Physics.

**Conclusion:** Design is stable and ready for Sign-off.
