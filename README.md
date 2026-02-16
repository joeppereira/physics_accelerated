# SerDes Cognitive Optimizer (SCO) - Repo B

## Core Purpose
Accelerated optimization of 128G PAM4 links using Physics-Informed Neural Operators. This repository consumes physical simulation data from **Repo A** (or dummy data) and uses it to train a high-speed surrogate model for sub-10ms design pathfinding.

## Tech Specs (3nm)
- **Node:** TSMC 3nm FinFET.
- **Rule:** DFE Tap-1 Limit = 35.0 mV.
- **Rule:** Thermal Ceiling = 105.0 °C.
- **Rule:** Target Efficiency < 0.60 pJ/bit.

## Skills Required (Gemini CLI)
- **Spectral AI:** Implementing FNO (Fourier Neural Operators).
- **Physical Handshaking:** Parquet metadata extraction and logic enforcement.
- **EDA Workflows:** PPA optimization and Yield analysis.

## Usage
1. **Setup:** `pip install -r requirements.txt`
2. **Data:** `python src/dummy_gen.py` (Generates physically-grounded data)
3. **Train:** `python src/train.py` (Physics-informed training)
4. **Optimize:** `python src/gepa.py` (Evolutionary Reflection Logic)
5. **Visualize:** `python src/visualizer.py`