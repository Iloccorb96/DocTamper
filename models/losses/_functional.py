import math
import numpy as np

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "focal_loss_with_logits",
    "softmax_focal_loss_with_logits",
    "soft_jaccard_score",
    "soft_dice_score",
    "wing_loss",
]


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def focal_loss_with_logits(
        output: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def softmax_focal_loss_with_logits(
        output: torch.Tensor,
        target: torch.Tensor,
        gamma: float = 2.0,
        reduction="mean",
        normalized=False,
        reduced_threshold: Optional[float] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Softmax version of focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of shape [B, C, *] (Similar to nn.CrossEntropyLoss)
        target: Tensor of shape [B, *] (Similar to nn.CrossEntropyLoss)
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    """
    log_softmax = F.log_softmax(output, dim=1)

    loss = F.nll_loss(log_softmax, target, reduction="none")
    pt = torch.exp(-loss)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * loss

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss = loss / norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def soft_jaccard_score(
        output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


def soft_dice_score(
        output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score


def soft_tversky_score(output: torch.Tensor, target: torch.Tensor, alpha: float, beta: float,
                       smooth: float = 0.0, eps: float = 1e-7, dims=None) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1. - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1. - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)
    return tversky_score


def wing_loss(output: torch.Tensor, target: torch.Tensor, width=5, curvature=0.5, reduction="mean"):
    """
    https://arxiv.org/pdf/1711.06753.pdf
    :param output:
    :param target:
    :param width:
    :param curvature:
    :param reduction:
    :return:
    """
    diff_abs = (target - output).abs()
    loss = diff_abs.clone()

    idx_smaller = diff_abs < width
    idx_bigger = diff_abs >= width

    loss[idx_smaller] = width * torch.log(1 + diff_abs[idx_smaller] / curvature)

    C = width - width * math.log(1 + width / curvature)
    loss[idx_bigger] = loss[idx_bigger] - C

    if reduction == "sum":
        loss = loss.sum()

    if reduction == "mean":
        loss = loss.mean()

    return loss


def label_smoothed_nll_loss(
        lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return: nll_loss 计算的是模型预测正确类别的概率的负对数。smooth_loss 是通过对lprobs中所有类别的对数概率求和，然后取负号得到的。这意味着它计算了所有类别的概率总和的对数。在标签平滑正则化中，这种方法可以鼓励模型对所有类别都有一个非零的预测概率，而不仅仅是最高概率的类别。

    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        #如果存在ignore_index，使用masked_fill将nll_loss和smooth_loss中对应于掩码位置的值设置为0，确保这些位置在损失计算中不会产生贡献。
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0) #
        nll_loss = -lprobs.gather(dim=dim, index=target)#使用gather函数从lprobs中根据target的索引提取对应的log-probability，计算出NLL（Negative Log-Likelihood）损失。dim=-1,最后一个维度，dim=1，通道维度.根据target中的label class info ，取出对应的位置的概率。
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)##计算了每个预测的对数概率之和。概率值在0-1之间，log_prob都是负的

        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)#smooth_loss是对lprobs在指定维度dim上求和后取负号得到的。这是标签平滑损失的平滑部分，它对所有类别的对数概率求和，然后从1中减去epsilon。
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)
    #损失归约:
    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / (lprobs.size(dim)-1)#epsilon=0.1，eps_i=0.05，在标签平滑中，我们不是将损失函数集中在正确的类别上（即概率为1），而是将一定量的损失分布到所有类别上。eps_i就是用来确定每个类别上平滑损失的权重。在这个例子中，每个错误类别（即不是模型预测最高概率的类别）将获得0.05的权重。
    loss = (1.0 - epsilon+eps_i) * nll_loss + eps_i * smooth_loss
    return loss
