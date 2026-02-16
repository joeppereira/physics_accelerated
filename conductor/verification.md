# Mutual Handshake & Verification

To verify the design across both repos, follow this Integration Test:

## Verification A: Data Integrity (Repo A → Repo B)
```bash
# In Repo A
python src/serdes_app_main.py --generate-training-data
# TEST: Check if data/samples_50k.parquet has 'dfe_tap1_limit_mv' in metadata.
```

## Verification B: Predictive Accuracy (Repo B → Repo A)
```bash
# In Repo B
python src/evaluation.py --model models/surrogate_v1.pth
# TEST: Ensure Tj error is < 2.0°C and Inference is < 10ms.
```

## Verification C: Usage as Library (Mutual)
Repo A can now optimize its link by importing the Bridge from Repo B:
```python
# Inside Repo A's optimizer.py
from physics_accelerated.bridge import SCO_Bridge

bridge = SCO_Bridge()
# Fast-predict 1000 combinations to find the one that hits 0.48 UI
best_settings = bridge.optimize_for_loss(-36.0) 
```
