import torch
import torch.nn.functional as F


def score_predictions(predictions, targets, loss_name, use_inversion):
    if torch.isnan(predictions).any():
        return -1e9, 1e9
    if loss_name == "mse":
        loss = F.mse_loss(predictions, targets).item()
    elif loss_name == "mae":
        loss = F.l1_loss(predictions, targets).item()
    else:
        loss = 1e9
    fitness = 1.0 / (loss + 1e-8) if use_inversion else -loss
    return fitness, loss
