import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.design_loader import DesignLoader
from src.physics_engine import VoxelThermalSolver3D

def evaluate_user_design():
    parser = argparse.ArgumentParser(description="Thermal Evaluation for User Designs")
    parser.add_argument("design_file", type=str, help="Path to JSON design file")
    parser.add_argument("--roi", type=str, default=None, help="ROI 'xmin,ymin,xmax,ymax'")
    args = parser.parse_args()
    
    # 1. Load Design & Stackup
    loader = DesignLoader()
    roi = [float(x) for x in args.roi.split(',')] if args.roi else None
    
    print(f"📂 Loading Design: {args.design_file}...")
    try:
        # Returns Power Grid AND Collapsed K-Values
        power_grid_l0, k_layers = loader.load_from_json(args.design_file, roi_bounds=roi)
    except Exception as e:
        print(f"❌ Error loading design: {e}")
        return

    # 2. Run Physics Solver (Ground Truth)
    print("⚙️ Running 3D FDM Physics Solver...")
    
    # Construct 5-Layer Power Volume (Power only on Layer 0)
    power_vol = np.zeros((5, 16, 16))
    power_vol[0] = power_grid_l0
    
    solver = VoxelThermalSolver3D(layers=5)
    temp_vol = solver.solve(power_vol, k_layers)
    
    # 3. Report
    peak_t = temp_vol[0].max()
    avg_t = temp_vol[0].mean()
    
    print("\n" + "="*40)
    print(f"🏆 THERMAL AUDIT REPORT")
    print("="*40)
    print(f"Peak Die Temp    : {peak_t:.1f} °C")
    print(f"Average Die Temp : {avg_t:.1f} °C")
    print("-" * 20)
    print(f"Stackup K_eff    : {[f'{k:.1f}' for k in k_layers]}")
    
    if peak_t > 105.0:
        print("Status: ❌ FAIL (Overheat)")
    else:
        print("Status: ✅ PASS")
        
    # 4. Viz
    plt.figure(figsize=(10, 8))
    plt.imshow(temp_vol[0], cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f"Thermal Map: {args.design_file}\n(Physically Solved with {len(k_layers)}-Layer Stack)")
    plt.savefig("plots/user_design_thermal.png")
    print("✅ Heatmap saved to plots/user_design_thermal.png")

if __name__ == "__main__":
    evaluate_user_design()