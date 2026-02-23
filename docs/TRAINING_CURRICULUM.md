# The "Flight Simulator" Curriculum: Chip Design Physics

This guide transforms the `physics_accelerated` platform into an educational tool. It is designed to teach **Intuition** about multi-physics interactions before you touch expensive commercial EDA licenses.

## 🎓 Philosophy: "Learn Fast, Sign-off Slow"
Commercial tools (Ansys, Cadence, Synopsys) are accurate but slow ($Hours/Run$).
This platform is approximate but instant ($Milliseconds/Run$).

**The Goal:** Use this tool to crash 1,000 chips in 1 minute, so you don't crash the real one in tape-out.

---

## 🧪 Module 1: Thermal Physics ("The Heat Trap")
**Objective:** Understand Power Density vs. Total Power.

### Exercise 1: The Pin-Hole Fire
Create a design with **1000mW** concentrated in a tiny spot vs. spread out.
1.  Edit `my_chip.json`: Set Block size to `100x100`. Power `1000`.
2.  Run: `python3 src/evaluate_design.py my_chip.json`
3.  **Result:** Observe Red Hotspot.
4.  **Fix:** Change size to `1000x1000`. Run again.
5.  **Lesson:** Area acts as a vertical heat pipe. Small area = High Resistance ($R_{th}$).

### Exercise 2: The Heat Sink Tax
Compare "Cheap Plastic" vs. "Expensive Copper".
1.  Run: `python3 src/analyze_package_tradeoff.py`
2.  **Observe:** The curve flattens.
3.  **Lesson:** You can trade Silicon Area ($$) for Package Cost ($$).

---

## ⚡ Module 2: Signal Integrity ("The Noisy Neighbor")
**Objective:** Understand Crosstalk and Isolation.

### Exercise 1: The Aggressor
Place a sensitive RX block right next to a noisy DSP.
1.  Edit `my_chip.json`: Set Spacing to `10um`.
2.  Run: `python3 src/evaluate_design.py` -> Check Margin.
3.  **Fix:** Move to `500um`.
4.  **Lesson:** Distance is a filter. Signal noise decays as $1/d^2$.

---

## ⏳ Module 3: Reliability ("The Time Machine")
**Objective:** Understand Aging (NBTI) and EOL (End of Life).

### Exercise 1: The Day-0 Trap
1.  Run `python3 src/analyze_spatial_aging.py`.
2.  Look at the "Year 0" line. It looks fine (Margin > 0).
3.  Look at the "Year 10" line. It collapses.
4.  **Lesson:** A chip that passes testing today might fail in 3 years if it runs hot. **Temperature drives Time.**

---

## 🌉 The Bridge to Industry
How skills learned here map to Enterprise Tools.

| Concept | Our Tool (Python) | Commercial Tool | The Industrial "Speak" |
| :--- | :--- | :--- | :--- |
| **Heat Map** | `physics_engine.py` (FDM) | **Ansys Icepak / Redhawk-SC** | "Check the thermal gradient across the macro." |
| **Optimization** | `gepa.py` (Evolutionary) | **Cadence Cerebrus / Synopsys DSO.ai** | "Run a DSE (Design Space Exploration) sweep." |
| **Aging** | `analyze_spatial_aging.py` | **RelXpert / Totem** | "What's the NBTI shift at EOL? Is it EM clean?" |
| **IR Drop** | `physics_engine_ir.py` | **Voltus / Redhawk** | "Do we have enough metal stripes to handle the current density?" |

## 🛠 Instructor Notes
To run this as a workshop:
1.  Have students fork the repo.
2.  Challenge: "Design a chip with >0.3 UI Margin using <8000 um² Area."
3.  Leaderboard: Rank solutions by **Area Efficiency**.
