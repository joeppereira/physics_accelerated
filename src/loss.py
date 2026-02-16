import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Combines MSE Data Loss with Physics Constraints.
    
    Components:
    1. Data Loss: MSE(y_pred, y_true)
    2. Constraint Loss: ReLU(|DFE_Tap1| - Limit)
    """
    def __init__(self, dfe_limit_mv=35.0, lambda_phy=0.1):
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.dfe_limit = dfe_limit_mv
        self.lambda_phy = lambda_phy
        
    def forward(self, pred, target, dfe_idx=-1):
        """
        pred: [batch, output_dim]
        target: [batch, output_dim]
        dfe_idx: Index of DFE Tap 1 in the output vector. Default -1 (last element).
        """
        # 1. Standard Regression Loss
        data_loss = self.mse(pred, target)
        
        # 2. Physics Constraint (Soft Penalty)
        # Extract DFE predictions
        dfe_pred_mv = pred[:, dfe_idx]
        
        # Penalty is 0 if |dfe| < limit, else (|dfe| - limit)^2
        # Using ReLU to activate only on violation
        violation = torch.relu(torch.abs(dfe_pred_mv) - self.dfe_limit)
        phy_loss = torch.mean(violation ** 2)
        
        total_loss = data_loss + (self.lambda_phy * phy_loss)
        
        return total_loss, data_loss, phy_loss
