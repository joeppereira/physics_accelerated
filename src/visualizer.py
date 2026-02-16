import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_yield_dashboard(data_path="data/samples_50k.parquet"):
    print(f"📊 Generating Yield Dashboard from {data_path}...")
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found.")
        return

    df = pd.read_parquet(data_path)
    
    # Ensure plots dir exists
    if not os.path.exists("plots"):
        os.makedirs("plots")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)

    # 1. Yield Bell Curve (Vertical Margin - Stage 6)
    # Using 'stage_6_eye_height_mv' as the main metric
    sns.histplot(df['stage_6_eye_height_mv'], kde=True, ax=axes[0, 0], color='blue')
    axes[0, 0].axvline(36.0, color='red', linestyle='--', label='Pass/Fail Spec (36mV)')
    axes[0, 0].set_title("Vertical Margin Yield Distribution (Stage 6)")
    axes[0, 0].set_xlabel("Eye Height (mV)")
    axes[0, 0].legend()

    # 2. Thermal Sensitivity (Tj vs Horizontal Margin - Stage 6)
    # Using 'stage_6_eye_width_ui'
    sample_df = df.sample(min(1000, len(df)))
    sns.scatterplot(x='max_junction_temp_c', y='stage_6_eye_width_ui', data=sample_df, 
                    ax=axes[0, 1], alpha=0.5, color='orange')
    axes[0, 1].set_title("Thermal Impact on Horizontal UI")
    axes[0, 1].set_xlabel("Junction Temperature (Tj) [°C]")
    axes[0, 1].set_ylabel("Eye Width (UI)")

    # 3. Phase Error Histogram (Simulated via Horizontal UI)
    sns.kdeplot(df['stage_6_eye_width_ui'], ax=axes[1, 0], fill=True, color='green')
    axes[1, 0].set_title("Horizontal Margin Distribution (CDR Accuracy)")
    axes[1, 0].axvline(0.48, color='red', linestyle='--', label='Spec (0.48 UI)')
    axes[1, 0].set_xlabel("Eye Width (UI)")
    axes[1, 0].legend()

    # 4. PPA Pareto (Power vs Yield)
    # Binning Power to see trend
    # Using 'total_power_mw'
    try:
        # Create bins for lineplot aggregation
        df['power_bins'] = pd.cut(df['total_power_mw'], bins=10)
        # We need the mid-point of bins for plotting if we want x-axis to be numeric, 
        # but seaborn accepts categorical x for lineplot often. 
        # Let's convert bins to string for categorical plot or use scatter for raw.
        # Following the snippet's intent with lineplot over bins.
        
        # Calculate mean yield per bin
        bin_stats = df.groupby('power_bins', observed=True)['stage_6_eye_height_mv'].mean().reset_index()
        bin_stats['power_mid'] = bin_stats['power_bins'].apply(lambda x: x.mid)
        
        sns.lineplot(x='power_mid', y='stage_6_eye_height_mv', data=bin_stats, ax=axes[1, 1], marker='o')
        axes[1, 1].set_title("Power-Performance Trade-off")
        axes[1, 1].set_xlabel("Power Consumption [mW]")
        axes[1, 1].set_ylabel("Mean Eye Height (mV)")
    except Exception as e:
        print(f"⚠️ Could not plot Power Pareto: {e}")

    plt.savefig("plots/yield_dashboard.png")
    print("✅ Yield Dashboard generated: plots/yield_dashboard.png")

if __name__ == "__main__":
    plot_yield_dashboard()
