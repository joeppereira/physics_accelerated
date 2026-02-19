import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.design_loader import DesignLoader
from src.bridge import OptimizerBridge

def evaluate_user_design():
    parser = argparse.ArgumentParser(description="AI Thermal Evaluation for User Designs")
    parser.add_argument("design_file", type=str, help="Path to JSON design file")
    parser.add_argument("--roi", type=str, default=None, help="ROI 'xmin,ymin,xmax,ymax'")
    args = parser.parse_args()
    
    loader = DesignLoader()
    roi = [float(x) for x in args.roi.split(',')] if args.roi else None
    
    print(f"📂 Loading Design: {args.design_file}...")
    try:
        # Load Power and Material Properties
        power_grid_l0, k_layers = loader.load_from_json(args.design_file, roi_bounds=roi)
    except Exception as e:
        print(f"❌ Error loading design: {e}")
        return

    # Run AI Inference
    print("🧠 Running Parametric Physics-NeMo Inference...")
    bridge = OptimizerBridge()
    temp_vol = bridge.predict_thermal_volume(power_grid_l0, k_layers)
    
    peak_t = temp_vol[0].max()
    avg_t = temp_vol[0].mean()
    
    print("\n" + "="*40)
    print(f"🏆 THERMAL AUDIT REPORT (AI PREDICTED)")
    print("="*40)
    print(f"Peak Die Temp    : {peak_t:.1f} °C")
    print(f"Average Die Temp : {avg_t:.1f} °C")
    print("-" * 20)
    print(f"Stackup K_eff    : {[f'{k:.1f}' for k in k_layers]}")
    
    if peak_t > 105.0:
        print("Status: ❌ FAIL (Overheat)")
    else:
        print("Status: ✅ PASS")
        
    plt.figure(figsize=(10, 8))
    plt.imshow(temp_vol[0], cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f"AI Thermal Map: {args.design_file}")
    plt.savefig("plots/user_design_thermal.png")
    print("✅ Heatmap saved to plots/user_design_thermal.png")

if __name__ == "__main__":
    evaluate_user_design()
