import pandas as pd
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS, NORM_FACTORS, unscale_output

def generate_sweep_data(output_path="simulated_result/sweep_analysis.csv"):
    print("📊 Generating 3D Trade-off Sweep Data (Aging Aware)...")
    
    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    try:
        model.load_state_dict(torch.load("models/surrogate_v1.pth", map_location=torch.device('cpu')))
    except:
        print("❌ Error: Model not trained.")
        return

    model.eval()
    
    bias_range = np.linspace(2, 15, 30) 
    area_range = np.linspace(5000, 15000, 30)
    
    results = []
    
    for bias in bias_range:
        for area in area_range:
            raw_input = {
                "ffe_m1": -0.05,
                "ffe_p1": -0.02,
                "area_um2": area,
                "bias_current_ma": bias,
                "rx_impedance_ohm": 100.0,
                "loss_db": -30.0,
                "temp_amb": 25.0
            }
            
            scaled_input = [
                raw_input["ffe_m1"],
                raw_input["ffe_p1"],
                raw_input["area_um2"] / NORM_FACTORS["area_um2"],
                raw_input["bias_current_ma"] / NORM_FACTORS["bias_current_ma"],
                raw_input["rx_impedance_ohm"] / NORM_FACTORS["rx_impedance_ohm"],
                raw_input["loss_db"] / NORM_FACTORS["loss_db"],
                raw_input["temp_amb"] / NORM_FACTORS["temp_amb"]
            ]
            
            with torch.no_grad():
                pred = model(torch.tensor(scaled_input).float().unsqueeze(0)).numpy()[0]
            
            # Unscale 5 outputs
            width, height, pwr, tj, life = unscale_output(pred)
            
            results.append({
                "bias_current_ma": bias,
                "area_um2": area,
                "eye_width_ui": width,
                "tj_c": tj,
                "lifetime_years": life
            })
            
    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Sweep data saved to {output_path}")

def plot_3d_tradeoff(data_path="simulated_result/sweep_analysis.csv"):
    if not os.path.exists(data_path):
        generate_sweep_data(data_path)

    df = pd.read_csv(data_path)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot: X=Bias, Y=Area, Z=Margin, Color=Lifetime
    img = ax.scatter(df['bias_current_ma'], df['area_um2'], df['eye_width_ui'], c=df['lifetime_years'], cmap='RdYlGn')
    
    ax.set_xlabel('Bias Current (mA)')
    ax.set_ylabel('Area (um2)')
    ax.set_zlabel('UI Margin')
    
    cbar = fig.colorbar(img, label='Lifetime (Years)')
    plt.title("3nm Architectural Trade-off: Reliability Surface")
    
    output_path = "plots/tradeoff_3d.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(output_path)
    print(f"📊 3D Trade-off Surface saved to {output_path}")

if __name__ == "__main__":
    plot_3d_tradeoff()
