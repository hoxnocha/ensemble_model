import torch 
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2,):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        
    def forward(self, inputs, targets):
        
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        F_loss = torch.mean(F_loss)
        
        
        return F_loss