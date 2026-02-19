import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.bridge import OptimizerBridge
from src.design_loader import DesignLoader

def evaluate_user_design():
    parser = argparse.ArgumentParser(description="AI Thermal Evaluation for User Designs")
    parser.add_argument("design_file", type=str, help="Path to JSON design file")
    parser.add_argument("--roi", type=str, default=None, help="ROI 'xmin,ymin,xmax,ymax' for zooming")
    args = parser.parse_args()
    
    # 1. Load Design
    loader = DesignLoader()
    roi = [float(x) for x in args.roi.split(',')] if args.roi else None
    
    print(f"📂 Loading Design: {args.design_file}...")
    try:
        power_grid = loader.load_from_json(args.design_file, roi_bounds=roi)
    except Exception as e:
        print(f"❌ Error loading design: {e}")
        return

    # 2. Run AI
    print("🧠 Running 3D Physics-NeMo Inference...")
    bridge = OptimizerBridge()
    temp_vol = bridge.predict_thermal_volume(power_grid)
    
    # 3. Report
    peak_t = temp_vol[0].max()
    avg_t = temp_vol[0].mean()
    
    print("\n" + "="*40)
    print(f"🏆 THERMAL AUDIT REPORT")
    print("="*40)
    print(f"Peak Temperature : {peak_t:.1f} °C")
    print(f"Average Temp     : {avg_t:.1f} °C")
    
    if peak_t > 105.0:
        print("Status: ❌ FAIL (Overheat)")
    else:
        print("Status: ✅ PASS")
        
    # 4. Viz
    plt.figure(figsize=(10, 8))
    plt.imshow(temp_vol[0], cmap='inferno', interpolation='nearest')
    plt.colorbar(label='Temperature (°C)')
    plt.title(f"Thermal Map: {args.design_file}\n(ROI Mode: {args.roi if args.roi else 'Full Die'})")
    plt.savefig("plots/user_design_thermal.png")
    print("✅ Heatmap saved to plots/user_design_thermal.png")

if __name__ == "__main__":
    evaluate_user_design()
