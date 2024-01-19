from typing import Callable

import schnetpack as spk
import schnetpack.nn as snn
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


def build_gated_equivariant_mlp(
    s_in: int,
    v_in: int,
    n_out: int,
    n_layers: int = 2,
    activation: Callable = F.silu,
    sactivation: Callable = F.silu,
):
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s instead of dense layers.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: Activation function for gating function.
        sactivation: Activation function for scalar outputs. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    s_neuron = s_in
    v_neuron = v_in
    s_neurons = []
    v_neurons = []
    for i in range(n_layers):
        s_neurons.append(s_neuron)
        v_neurons.append(v_neuron)
        s_neuron = max(n_out, s_neuron // 2)
        v_neuron = max(n_out, v_neuron // 2)
    s_neurons.append(n_out)
    v_neurons.append(n_out)

    n_gating_hidden = s_neurons[:-1]

    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        snn.GatedEquivariantBlock(
            n_sin=s_neurons[i],
            n_vin=v_neurons[i],
            n_sout=s_neurons[i + 1],
            n_vout=v_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        snn.GatedEquivariantBlock(
            n_sin=s_neurons[-2],
            n_vin=v_neurons[-2],
            n_sout=s_neurons[-1],
            n_vout=v_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net


class ConditionedScoreModel(LightningModule):
    def __init__(
        self, representation, time_dim=2, conditioning=None, gated_blocks=3, **kwargs
    ):
        super().__init__(**kwargs)
        self.representation = representation
        self.time_dim = time_dim
        self.omega = 2 * torch.pi
        self.cond_dim = conditioning.dim if conditioning is not None else 0
        self.net = build_gated_equivariant_mlp(
            self.representation.embedding.dim + time_dim + self.cond_dim,
            self.representation.embedding.dim,
            1,
            n_layers=gated_blocks,
        )
        self.conditioning = conditioning

    def forward(self, batch, t=None, prob=0.0, condition=None):
        if (
            "scalar_representation" not in batch
            and "vector_representation" not in batch
        ):
            inputs = self.representation(batch)
        else:
            inputs = batch

        scalar_representation = inputs["scalar_representation"]

        if t is None:
            time_cond = torch.zeros(
                (scalar_representation.shape[0], self.time_dim), device=self.device
            )
        else:
            time_cond = torch.concatenate(
                (torch.sin(self.omega * t), torch.cos(self.omega * t)), dim=-1
            )

        scalar_representation = torch.cat((scalar_representation, time_cond), dim=-1)

        if self.conditioning is not None:
            cond = self.conditioning(batch, prob=prob, condition=condition)
            scalar_representation = torch.cat((scalar_representation, cond), dim=-1)

        vector_representation = inputs["vector_representation"]

        scalar, vector = self.net([scalar_representation, vector_representation])
        return vector


class ScoreModel(LightningModule):
    def __init__(self, representation, **kwargs):
        super().__init__(**kwargs)
        self.representation = representation
        self.net = spk.nn.blocks.build_gated_equivariant_mlp(
            self.representation.embedding.dim, 1
        )

    def forward(self, batch, t=None, prob=0.0, condition=None):
        inputs = self.representation(batch, t=t, prob=prob, condition=condition)
        scalar_representation = inputs["scalar_representation"]
        vector_representation = inputs["vector_representation"]

        scalar, vector = self.net([scalar_representation, vector_representation])
        return vector
