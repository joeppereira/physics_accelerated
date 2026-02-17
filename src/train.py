import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import argparse
from src.surrogate import MiniSAUFNOJEPA
from src.schema import FEATURES, TARGETS

def stable_loss(pred, target):
    """
    Weighted MSE Loss with Physics Constraints.
    Targets: [Width, Height, Power, Tj, Lifetime]
    Weights: [50.0, 1.0, 1.0, 1.0, 5.0]
    """
    # Dynamic weights based on target length
    if pred.shape[1] == 5:
        # Lifetime included
        weights = torch.tensor([50.0, 1.0, 1.0, 1.0, 5.0]).to(pred.device)
    else:
        # Fallback for old model
        weights = torch.tensor([50.0, 1.0, 1.0, 1.0]).to(pred.device)
        
    loss = torch.mean(weights * (pred - target)**2)
    
    # Physics Constraint: Tj (index 3) cannot be below Ambient (0.2 normalized)
    tj_violation = torch.relu(0.2 - pred[:, 3]).mean()
    
    return loss + (10.0 * tj_violation)

def train_model(data_path="data/samples_normalized.parquet", epochs=50):
    print(f"🚀 Starting Aging-Aware Training ({epochs} epochs)...")
    
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
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/surrogate_v1.pth")
    print("✅ Training complete. Model saved to models/surrogate_v1.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs)