import torch.nn as nn
import copy

class EMA(nn.Module):
    def __init__(self, model, beta=0.99):
        super(EMA, self).__init__()
        self.beta = beta

        self.ema_model = copy.deepcopy(model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()

    def update(self, model):
        for p, p_ema in zip(model.parameters(), self.ema_model.parameters()):
            p_ema.data.lerp_(p, weight=(1.0 - self.beta))
