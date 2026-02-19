import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from src.surrogate import PhysicsNeMoFNO2D

def train_spatial_model(epochs=50):
    print(f"🚀 Starting 2D FNO Spatial Training ({epochs} epochs)...")
    
    x, y = None, None
    
    # 1. Try External Super-Res (64x64)
    if os.path.exists("../serdes_architect/data/x_64.pt"):
        print("   -> Loading External 64x64 Super-Res Data")
        x = torch.load("../serdes_architect/data/x_64.pt")
        y = torch.load("../serdes_architect/data/y_64.pt")
        
    # 2. Try Local Parametric (16x16)
    elif os.path.exists("data/x_parametric.pt"):
        print("   -> Loading Local Parametric Data")
        x = torch.load("data/x_parametric.pt")
        y = torch.load("data/y_parametric.pt")
        
    # 3. Try Local 3D (16x16)
    elif os.path.exists("data/x_3d.pt"):
        print("   -> Loading Local 3D Data")
        x = torch.load("data/x_3d.pt")
        y = torch.load("data/y_3d.pt")
        
    if x is None:
        print("❌ Error: No training data found.")
        return

    model = PhysicsNeMoFNO2D()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/spatial_fno_v1.pth")
    print("✅ Training complete. 2D Surrogate saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    train_spatial_model(epochs=args.epochs)
