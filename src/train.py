import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import argparse
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS

def stable_loss(pred, target):
    # Targets: [Margin, Pwr, Trx, Tdsp, Life]
    weights = torch.tensor([100.0, 1.0, 1.0, 1.0, 5.0]).to(pred.device)
    loss = torch.mean(weights * (pred - target)**2)
    return loss

def train_model(data_path="data/samples_normalized.parquet", epochs=50):
    print(f"🚀 Starting Spatial-Aware Training ({epochs} epochs)...")
    
    if not os.path.exists(data_path):
        print("❌ Error: Training data missing.")
        return

    df = pd.read_parquet(data_path)
    X = torch.tensor(df[FEATURES].values).float()
    y = torch.tensor(df[TARGETS].values).float()

    model = MiniSAUFNOJEPA(in_dim=len(FEATURES), out_dim=len(TARGETS))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = stable_loss(output, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/surrogate_v1.pth")
    print("✅ Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train_model(epochs=args.epochs)
