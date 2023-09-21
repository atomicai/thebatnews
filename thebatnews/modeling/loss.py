from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from dotmap import DotMap


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, reduction="none", ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        balanced_weights = (
            nn.Parameter(torch.tensor(class_weights), requires_grad=False) if class_weights is not None else None
        )
        self.fct = nn.CrossEntropyLoss(
            weight=balanced_weights,
            reduction=reduction,
            ignore_index=ignore_index,
        )

    def forward(self, x, target):
        return self.fct(x, target)


class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2.0, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, x, target):
        """
        x: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(x, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, reduction="none")
        return loss


class TverskyLoss(nn.Module):
    r"""Criterion that computes Tversky Coeficient loss.
    According to [1], we compute the Tversky Coefficient as follows:
    .. math::
        \text{S}(P, G, \alpha; \beta) =
          \frac{|PG|}{|PG| + \alpha |P \ G| + \beta |G \ P|}
    where:
       - :math:`P` and :math:`G` are the predicted and ground truth binary
         labels.
       - :math:`\alpha` and :math:`\beta` control the magnitude of the
         penalties for FPs and FNs, respectively.
    Notes:
       - :math:`\alpha = \beta = 0.5` => dice coeff
       - :math:`\alpha = \beta = 1` => tanimoto coeff
       - :math:`\alpha + \beta = 1` => F beta coeff
    Shape:
        - Input: :math:`(N, C)` where C = number of classes.
        - Target: :math:`(N,)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> loss = TverskyLoss(alpha=0.5, beta=0.5)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        gamma: int = 0,
        scale: float = 1.0,
        reduction: Optional[str] = "mean",
        ignore_index: int = -100,
        eps: float = 1e-6,
        smooth: float = 0,
    ) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: int = gamma
        self.scale: float = scale
        self.reduction: Optional[str] = reduction
        self.ignore_index: int = ignore_index
        self.eps: float = eps
        self.smooth: float = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(input.shape) == 2:
            if input.shape[0] != target.shape[0]:
                raise ValueError(
                    "number of elements in input and target shapes must be the same. Got: {}".format(
                        input.shape, input.shape
                    )
                )
        else:
            raise ValueError("Invalid input shape, we expect or NxC. Got: {}".format(input.shape))
        if not input.device == target.device:
            raise ValueError("input and target must be in the same device. Got: {}".format(input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        input_soft = self.scale * ((1 - input_soft) ** self.gamma) * input_soft

        # filter labels
        target = target.type(torch.long)
        input_mask = target != self.ignore_index

        target = target[input_mask]
        input_soft = input_soft[input_mask]

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).to(input.device).type(input_soft.dtype)

        # compute the actual dice score
        intersection = torch.sum(input_soft * target_one_hot, -1)
        fps = torch.sum(input_soft * (1.0 - target_one_hot), -1)
        fns = torch.sum((1.0 - input_soft) * target_one_hot, -1)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = (numerator + self.smooth) / (denominator + self.eps + self.smooth)
        tversky_loss = 1.0 - tversky_loss

        if self.reduction is None or self.reduction == "none":
            return tversky_loss
        elif self.reduction == "mean":
            return torch.mean(tversky_loss)
        elif self.reduction == "sum":
            return torch.sum(tversky_loss)
        else:
            raise NotImplementedError("Invalid reduction mode: {}".format(self.reduction))


class AngularPenaltySMLoss(nn.Module):
    def __init__(self, in_features, out_features, loss_type="arcface", eps=1e-7, s=None, m=None):
        """
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        """
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ["arcface", "sphereface", "cosface"]
        if loss_type == "arcface":
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == "sphereface":
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == "cosface":
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        """
        input shape (N, in_features)
        """
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == "cosface":
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == "arcface":
            numerator = self.s * torch.cos(
                torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
                + self.m
            )
        if self.loss_type == "sphereface":
            numerator = self.s * torch.cos(
                self.m
                * torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.0 + self.eps, 1 - self.eps))
            )

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


Losses = DotMap(
    {
        "CrossEntropyLoss": CrossEntropyLoss,
        "FocalLoss": FocalLoss,
        "TverskyLoss": TverskyLoss,
        "AngularPenaltySMLoss": AngularPenaltySMLoss,
    }
)


__all__ = ["Losses"]
