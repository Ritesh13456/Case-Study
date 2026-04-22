import torch
from model import PrunableLinear

def compute_sparsity_loss(model):
    loss = 0
    total_params = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores / 0.5) 
            loss += gates.sum()
            total_params += gates.numel()

    return loss / total_params  


def compute_sparsity(model, threshold=1e-2):
    total = 0
    pruned = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores / 0.5)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()

    return 100 * pruned / total