# src/strategies/uncertainty.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional
from src.utils.device import device


@torch.no_grad()
def entropy_from_logits(logits: torch.Tensor, eps=1e-8) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + eps)).sum(dim=-1)


@torch.no_grad()
def per_sample_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = model(x.to(device))
    loss = F.cross_entropy(logits, y.to(device), reduction="none")
    return loss.cpu()


@torch.no_grad()
def per_sample_entropy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = model(x.to(device))
    return entropy_from_logits(logits).cpu()


@torch.no_grad()
def bald_information_gain(
    model: nn.Module,
    x: torch.Tensor,
    mc: int = 5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    IG = H(E[p]) - E[H(p)]
    Uses MC Dropout to approximate epistemic uncertainty.
    """

    was_training = model.training
    model.train()  # enable dropout

    probs = []
    xd = x.to(device)

    for _ in range(mc):
        logits = model(xd)
        probs.append(F.softmax(logits, dim=-1))

    p = torch.stack(probs, dim=0)     # (T,B,C)
    p_mean = p.mean(dim=0)            # (B,C)

    H_mean = -(p_mean * torch.log(p_mean + eps)).sum(dim=-1)
    H_exp  = -((p * torch.log(p + eps)).sum(dim=-1)).mean(dim=0)

    # restore mode properly
    if not was_training:
        model.eval()

    return (H_mean - H_exp).cpu()


def refresh_scores(
    model: nn.Module,
    buf,
    score_mode: str,
    refresh_k: int = 256,
    mc: int = 5,
    beta: float = 0.9
):
    if len(buf) == 0:
        return

    idx = random.sample(range(len(buf.items)), k=min(refresh_k, len(buf.items)))

    x = torch.stack([buf.items[i].x for i in idx])
    y = torch.tensor([buf.items[i].y for i in idx], dtype=torch.long)

    if score_mode == "loss":
        s = per_sample_loss(model, x, y)

    elif score_mode == "entropy":
        s = per_sample_entropy(model, x)

    elif score_mode == "ig":
        IG = bald_information_gain(model, x, mc=mc)
        L  = per_sample_loss(model, x, y)

        eps = 1e-8
        IGn = (IG - IG.min()) / (IG.max() - IG.min() + eps)
        Ln  = (L  - L.min())  / (L.max()  - L.min()  + eps)

        # hybrid prioritization (IMPORTANT)
        s = beta * IGn + (1 - beta) * Ln

    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    # normalize to [0,1]
    eps = 1e-8
    sn = (s - s.min()) / (s.max() - s.min() + eps)

    for j, i in enumerate(idx):
        buf.items[i].score = float(sn[j].item()) + 1e-6

@torch.no_grad()
def bald_information_gain(
    model: nn.Module,
    x: torch.Tensor,
    mc: int = 5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Approximates BALD (Information Gain) using MC Dropout.

    IG = H(E[p(y|x, θ)]) - E[H(p(y|x, θ))]
    """

    # Save original mode
    was_training = model.training

    # Enable dropout for MC sampling
    model.train()

    probs = []
    xd = x.to(device)

    for _ in range(mc):
        logits = model(xd)
        probs.append(F.softmax(logits, dim=-1))

    p = torch.stack(probs, dim=0)   # (T, B, C)
    p_mean = p.mean(dim=0)          # (B, C)

    H_mean = -(p_mean * torch.log(p_mean + eps)).sum(dim=-1)
    H_exp  = -((p * torch.log(p + eps)).sum(dim=-1)).mean(dim=0)

    # Restore original mode properly
    if not was_training:
        model.eval()

    return (H_mean - H_exp).cpu()
