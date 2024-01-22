import torch
import schnetpack as spk
from schnetpack import properties
from typing import Dict



class ConditionalPaiNN(spk.representation.PaiNN):
    def __init__(self, n_interactions, radial_basis, embedding, max_z=100, **kwargs):
        super().__init__(
            embedding.dim, n_interactions, radial_basis, max_z=max_z, **kwargs
        )
        self.embedding = embedding

    def forward(
        self, inputs: Dict[str, torch.Tensor], t=None, prob=0.0, condition=None
    ):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        atomic_numbers = inputs[properties.Z]
        r_ij = inputs[properties.Rij]
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]
        n_atoms = atomic_numbers.shape[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        phi_ij = self.radial_basis(d_ij)
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        q = self.embedding(inputs, t=t, prob=prob, condition=condition)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)

        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu
        return inputs
