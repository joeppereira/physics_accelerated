import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from src.surrogate import PhysicsNeMoFNO2D

def train_spatial_model(epochs=50):
    print(f"🚀 Starting 2D FNO Spatial Training ({epochs} epochs)...")
    
    # Load Voxel Data from External Physics Factory
    try:
        x = torch.load("../serdes_architect/data/x_3d.pt")
        y = torch.load("../serdes_architect/data/y_3d.pt")
    except:
        print("❌ Error: External 3D data missing. Run src/data_gen.py in serdes_architect.")
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