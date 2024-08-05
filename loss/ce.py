import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100):
        """
        CrossEntropyLoss module
        
        Args:
            weight (Tensor, optional): a manual rescaling weight given to each class.
            reduction (str, optional): specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            ignore_index (int, optional): Specifies a target value that is ignored and does not contribute to the input gradient.
        """
        super(CrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CrossEntropyLoss
        
        Args:
            y_pred (torch.Tensor): Predicted logits (N, C, H, W)
            y_true (torch.Tensor): Ground truth labels (N, H, W)
            
        Returns:
            torch.Tensor: Computed cross entropy loss
        """
        y_pred = y_pred.float()
        y_true = y_true.long()
        return F.cross_entropy(
            y_pred, 
            y_true, 
            weight=self.weight, 
            reduction=self.reduction, 
            ignore_index=self.ignore_index
        )
