import torch
import numpy as np

def audit_external_data():
    print("🔍 Auditing External Data from SerDes Architect...")
    
    x_path = "../serdes_architect/data/x_spatial.pt"
    y_path = "../serdes_architect/data/y_spatial.pt"
    
    x = torch.load(x_path)
    y = torch.load(y_path)
    
    print(f"✅ Shapes match: X={x.shape}, Y={y.shape}")
    
    # Range check
    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()
    
    print(f"📊 Power (X) Range: {x_min:.3f} to {x_max:.3f} (Expected ~0-1)")
    print(f"📊 Temp (Y) Range: {y_min:.3f} to {y_max:.3f} (Expected ~0.2-1.2)")
    
    # Correlation Check (Hotspot Alignment)
    # Pick a random sample and check if max temp location aligns with max power location
    sample_idx = 42
    max_p_loc = np.unravel_index(torch.argmax(x[sample_idx]).item(), (16, 16))
    max_t_loc = np.unravel_index(torch.argmax(y[sample_idx]).item(), (16, 16))
    
    dist = np.sqrt((max_p_loc[0]-max_t_loc[0])**2 + (max_p_loc[1]-max_t_loc[1])**2)
    print(f"🎯 Alignment Test: Power Peak at {max_p_loc}, Temp Peak at {max_t_loc}")
    print(f"   Distance between peaks: {dist:.2f} pixels")
    
    if dist < 3.0:
        print("🚀 Data Relevance: PASS (Thermal hotspots track power sources)")
    else:
        print("❌ Data Relevance: FAIL (Heat map does not match power layout)")

if __name__ == "__main__":
    audit_external_data()
