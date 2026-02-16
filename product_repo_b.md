# Product Specification: Repo B (physics_accelerated)

This document defines the "Intellectual Property" of the AI repo. It treats Repo A as the "Sensor" and Repo B as the "Brain."

| Feature | Specification |
| :--- | :--- |
| Commercial Purpose | |
| Model Architecture | Mini-SAUFNO-JEPA (~0.6M Params) 100x faster than Repo A's physics. |
| Input Interface | 10-Scalar Config (FFE, Loss, Temp) Matches Repo A's simulation_result schema. |
| Output Contract | 15-Scalar Response (7-Stage Margins, Tj) Predicts the full 7-stage handshake. |
| Logic Constraint | Physics-Informed Loss (PIL) Enforces the 35mV DFE Tap-1 limit from 3nm spec. |
| Optimization | GEPA (Evolutionary Reflection) Evolves PPA to hit < 0.60 pJ/bit. |
