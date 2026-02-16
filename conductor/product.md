# Product Definition: physics_accelerated

This is the "Brain" of the AI repo. It handles the SAUFNO-JEPA surrogate and GEPA evolution.

## Core Features
- **Model**: Mini-SAUFNO-JEPA (~0.6M Params)
- **Performance**: 100x faster than physical simulation.
- **Constraints**: Physics-Informed Loss (PIL) for 35mV DFE Tap-1 limit.
- **Optimization**: GEPA (Evolutionary Reflection) for < 0.60 pJ/bit.
