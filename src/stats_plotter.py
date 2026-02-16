import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_thermal_sensitivity(data_path="data/samples_50k.parquet"):
    if not os.path.exists(data_path):
        print(f"❌ Error: Data not found at {data_path}. Run dummy_gen.py first.")
        return

    df = pd.read_parquet(data_path)
    
    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)
    
    # 1. Thermal Sensitivity Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='tj_c', y='eye_width_ui', data=df.sample(min(2000, len(df))), hue='pwr', palette='viridis')
    plt.axhline(0.48, color='red', linestyle='--', label='UI Spec Limit')
    plt.title("Cognitive Insight: Horizontal Margin vs. Junction Temp")
    plt.xlabel("Junction Temperature (Tj) [°C]")
    plt.ylabel("Horizontal Eye [UI]")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/thermal_sensitivity.png")
    
    # 2. Yield Bell Curve
    plt.figure(figsize=(10, 6))
    sns.histplot(df['eye_height_mv'], kde=True, color='blue')
    plt.axvline(36.0, color='red', linestyle='--', label='36mV Spec')
    plt.title("Predicted Design Yield (3nm Process)")
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/yield_curve.png")
    
    print("✅ Yield Sensitivity visuals saved to plots/")

if __name__ == "__main__":
    plot_thermal_sensitivity()
