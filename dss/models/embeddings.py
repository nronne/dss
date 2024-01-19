import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from schnetpack import properties

class NodeEmbedding(LightningModule):
    def __init__(self, type_dim=32, time_dim=2, conditioning=None):
        super().__init__()
        self.type_embedding = nn.Embedding(
            100,
            type_dim,
            padding_idx=0,
        )
        self.time_dim = time_dim

        self.conditioning = conditioning
        if conditioning is None:
            self.cond_dim = 0
        else:
            self.cond_dim = conditioning.dim

        self.omega = 2 * torch.pi

        self.dim = type_dim + time_dim + self.cond_dim

    def forward(self, inputs, t=None, prob=0.0, condition=None):
        atomic_numbers = inputs[properties.Z]
        h0 = self.type_embedding(atomic_numbers)
        if t is None:
            h1 = torch.zeros((h0.shape[0], self.time_dim), device=self.device)
        else:
            h1 = torch.concatenate(
                (torch.sin(self.omega * t), torch.cos(self.omega * t)), dim=-1
            )

        if self.conditioning is not None:
            h2 = self.conditioning(inputs, prob=prob, condition=condition)
            h = torch.cat((h0, h1, h2), dim=-1)

        else:
            h = torch.cat((h0, h1), dim=-1)

        return h
    
class Conditioning(LightningModule):
    def __init__(self, dim, key="energy", tau=1, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.key = key
        self.tau = tau

    def forward(self, inputs, prob=0.0, condition=None):
        """
        prob: probability of conditioning on energy ie. prob=0 means no conditioning
        """
        if prob <= 0.0:
            return torch.zeros(
                (inputs[properties.R].shape[0], self.dim), device=self.device
            )

        if condition is None:
            x = inputs[self.key]
        else:
            x = condition

        p = torch.rand(x.shape[0], device=self.device)
        null = p > prob

        # expand to all atoms
        x /= self.tau
        x = x[inputs["_idx_m"]]
        h = torch.stack((x, torch.exp(x)), dim=-1)  # torch.exp(x)
        null = null[inputs["_idx_m"]]

        h[null, :] = torch.zeros(h.shape[1], device=self.device, dtype=h.dtype)
        return h    
