# 128G SerDes Cognitive Optimization Platform

This platform integrates a **Physics-Informed Ground Truth Engine** with an **AI Cognitive Optimizer** to accelerate 128G PAM4 design pathfinding.

## Key Documentation
- **[Developer Onboarding](Developer_Onboarding.md):** The primary entry point for new developers.
- **[Executive Sign-off Report](Project_Signoff_Template.md):** The final architectural verdict template.
- **[Technical Architecture & Developer Guide](TECHNICAL_ARCHITECTURE.md):** Deep dive into the physics loops, thermal taxes, and AI loss functions.

## Directory Structure
- `physics_core/`: The "Ground Truth" Engine (Simulated Physics).
- `cognitive_optimizer/`: The AI Acceleration Layer (Surrogate Models).
- `reports/`: Automated Sign-off PDFs and JSONs.
- `run_full_cycle.sh`: The Master Orchestrator script.

## Getting Started

### 1. Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Hardware Acceleration (Apple Silicon M-Series)
If you are running on an M4 (or M1/M2/M3) chip, ensure you have the `mps` (Metal Performance Shaders) backend enabled in PyTorch for accelerated training.
```bash
# Verify MPS availability
python3 -c "import torch; print(torch.backends.mps.is_available())"
```
*Note: The script automatically detects `mps` if available.*

### 3. Run the Platform
To run the full design cycle (Physics -> Train -> Optimize -> Report):
```bash
bash run_full_cycle.sh
```
