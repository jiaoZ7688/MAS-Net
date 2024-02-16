from torch.nn import functional as F
import torch.nn as nn
import torch
from torch.nn.modules.loss import _Loss

# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):

#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       

#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()                            
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss

#         return Dice_BCE


class DiceBCELoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, y_pred, y_true):

        y_pred = y_pred.sigmoid()

        bs = y_true.size(0)
        ch = y_true.size(1)
        dims = (0 , 2)
        y_pred = y_pred.view(bs, ch, -1)
        y_true = y_true.view(bs, ch, -1)

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        pass

def soft_tversky_score(
    output,
    target,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
):
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)

    return tversky_score

class TverskyLoss(DiceBCELoss):
    """Tversky loss for image segmentation task.
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases

    Args:
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Positives)
        gamma: Constant that squares the error function. Defaults to ``1.0``

    Return:
        loss: torch.Tensor

    """
    def __init__(
        self,
        smooth: float = 0.0,
        eps: float = 1e-7,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.0,
    ):
        super().__init__(smooth, eps)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def aggregate_loss(self, loss):
        return loss.mean() ** self.gamma

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None):
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)
