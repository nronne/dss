import pytorch_lightning as pl
import torch
import torch.nn as nn
from dss.utils import TruncatedNormal
from schnetpack import properties  # change to use this at some point
from schnetpack.atomistic import PairwiseDistances
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import WrapPositions

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]


class VPDiffusion(pl.LightningModule):
    def __init__(
        self,
        score_model,
        neighbour_list,
        potential_model=None,
        pairwise_distance=PairwiseDistances(),
        potential_head=True,
        beta_min=1e-2,
        beta_max=3,
        eps=1e-5,
        condition_config={
            "train_prob": 0.5,
        },
        loss_config={
            "energy_fn": nn.MSELoss(),
            "forces_fn": nn.MSELoss(),
            "energy_weight": 0.01,
            "forces_weight": 0.99,
        },
        optim_config={"lr": 1e-3},
        scheduler_config={"factor": 0.05, "patience": 20},
    ):
        super(VPDiffusion, self).__init__()
        self.score_model = score_model
        self.neighbour_list = neighbour_list
        self.pairwise_distance = pairwise_distance

        self.wrap_positions = WrapPositions()
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(beta_max)
        self.eps = torch.tensor(eps)

        self.potential_model = potential_model
        self.loss_config = loss_config

        if self.score_model.representation.embedding.cond_dim > 0:
            self.train_prob = condition_config.get("train_prob", 0.5)
        else:
            self.train_prob = 0.0

        self.condition_config = condition_config
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config

    def setup(self, stage=None):
        self.offsets = torch.tensor(OFFSET_LIST).float().to(self.device)

    def beta_t(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t):
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def forward_drift(self, x, t):
        return -0.5 * self.beta_t(t) * x

    def dispersion(self, t):
        return torch.sqrt(self.beta_t(t))

    def mean_factor(self, t):
        return torch.exp(-self.alpha_t(t))

    def marginal_probability(self, t):
        return 1 - torch.exp(-self.alpha_t(t))

    # LightningModule methods

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            self.log("train_" + k, v)
        return losses["loss"]

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        if self.potential_model is not None:
            torch.set_grad_enabled(True)

        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            self.log("val_" + k, v)
        return losses["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_config
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def forward(self, batch, t, prob=0.0, condition=None):
        if (
            "scalar_representation" not in batch
            and "vector_representation" not in batch
        ):
            batch = self.preprocess_batch(batch)
        v = self.score_model(batch, t, prob=prob, condition=condition).view(
            batch[properties.R].shape
        )
        return v

    def sample_forward(self, batch, t, w=None, condition=None):
        # if self.potential_model is not None:
        #     self.potential_model.initialize_derivatives(batch)

        if w is None:
            return self.forward(batch, t)
        else:
            if (
                "scalar_representation" not in batch
                and "vector_representation" not in batch
            ):
                batch = self.preprocess_batch(batch)

            s0 = self.score_model(batch, t, prob=1.0, condition=condition).view(
                batch[properties.R].shape
            )
            s1 = self.score_model(batch, t).view(batch[properties.R].shape)

            s = (1 + w) * s0 - w * s1

            return s

    @torch.set_grad_enabled(True)
    def potential(self, batch):
        batch = self.preprocess_batch(batch, save_keys=[])
        self.potential_model(batch)
        return batch

    def batch_wrap(self, batch):  # Make decorator for batching functions like this
        batch_list = self._split_batch(batch)
        for i in range(len(batch_list)):
            batch_list[i] = self.wrap_positions(batch_list[i])

        batch = self._collate_batch(batch_list)
        return batch

    def periodic_distance(self, X, N, cell):  # TODO
        """
        TODO
        X: (N, 3)
        N: (N, 3)
        cell: (3, 3)

        Takes X and N (noise) and computes the minimum distance between X and Y=X+N
        taking into account periodic boundary conditions.
        """
        cell_offsets = torch.matmul(self.offsets, cell)
        Y = X + N

        # Compute distances between X and Y + cell_offsets
        Y = Y.unsqueeze(1)
        cell_offsets = cell_offsets.unsqueeze(0)
        Y = Y + cell_offsets
        distances = torch.norm(X.unsqueeze(1) - Y, dim=2)

        argmin_distances = torch.argmin(distances, dim=1)
        # find the Y that minimizes the distance
        Y = Y[torch.arange(Y.shape[0]), argmin_distances]
        min_N = Y - X

        return min_N

    def batch_clone(self, batch, ignore=()):
        batch_copy = {}
        for key in batch.keys():
            if key not in ignore:
                batch_copy[key] = batch[key].clone()

        return batch_copy

    def _split_batch(self, batch):
        atom_types = batch[properties.Z]
        positions = batch[properties.R]
        n_atoms = batch[properties.n_atoms]
        idx_m = batch[properties.idx_m]
        cells = batch[properties.cell]
        pbc = batch[properties.pbc]

        mask = batch.get("mask", torch.zeros_like(positions, dtype=torch.bool))

        n_structures = n_atoms.shape[0]

        z_confinement = batch.get("z_confinement", None)

        if z_confinement is not None:
            z_confinement = z_confinement.view(n_structures, 2)

        output_list = []
        idx_c = 0
        for idx in range(n_structures):
            curr_n_atoms = n_atoms[idx]
            inputs = {
                properties.n_atoms: torch.tensor([curr_n_atoms]),
                properties.Z: atom_types[idx_c : idx_c + curr_n_atoms],
                properties.R: positions[idx_c : idx_c + curr_n_atoms],
                "mask": mask[idx_c : idx_c + curr_n_atoms],
            }

            if cells is None:
                inputs[properties.cell] = None
                inputs[properties.pbc] = None
            else:
                inputs[properties.cell] = cells[idx][None, :, :]
                inputs[properties.pbc] = pbc[idx][None]

            if z_confinement is not None:
                inputs["z_confinement"] = z_confinement[idx]

            idx_c += curr_n_atoms
            output_list.append(inputs)

        return output_list

    def _collate_batch(self, batch_list):
        return _atoms_collate_fn(batch_list)

    def _random_positions(self, structure):
        positions = structure[properties.R]
        cell = structure[properties.cell].clone().view(3, 3)
        corner = torch.zeros(3)

        if structure.get("z_confinement", None) is not None:
            cell[2, 2] = structure["z_confinement"][1] - structure["z_confinement"][0]
            corner[2] = structure["z_confinement"][0]

        mask = structure.get(
            "mask", torch.zeros_like(positions, dtype=torch.bool)
        ).bool()

        f = torch.rand_like(positions)
        random_positions = torch.matmul(f, cell) + corner
        structure[properties.R][~mask] = random_positions[~mask]

        return structure

    def batch_random_positions(self, batch):
        batch_list = self._split_batch(batch)
        for i in range(len(batch_list)):
            batch_list[i] = self._random_positions(batch_list[i])

        batch = self._collate_batch(batch_list)
        return batch

    def preprocess_batch(self, batch, save_keys=["energy", "forces"]):
        saved = {}
        for key in save_keys:
            if key in batch:
                saved[key] = batch[key]

        batch_list = self._split_batch(batch)
        for i in range(len(batch_list)):
            batch_list[i] = self.neighbour_list(batch_list[i])
            batch_list[i] = self.pairwise_distance(batch_list[i])

        batch = self._collate_batch(batch_list)

        for key in saved:
            batch[key] = saved[key]

        return batch

    def loss(self, batch, batch_idx):
        ### score matching loss
        noised_batch = self.batch_clone(batch)
        mask = batch.get(
            "mask", torch.zeros_like(batch[properties.R], dtype=torch.bool)
        ).bool()

        B = len(batch["_idx"])
        t = torch.rand((B, 1), device=self.device) * (1 - self.eps) + self.eps
        t = t[batch["_idx_m"]]

        vars = self.marginal_probability(t)
        stds = torch.sqrt(vars)
        noise = torch.randn_like(batch["_positions"], device=self.device)
        # mask noise
        noise[mask] = 0
        t[mask.sum(dim=1) != 0] = 0  # setting t to 0 for masked atoms

        if batch.get("z_confinement", None) is not None:
            z_confinement = batch["z_confinement"].view(B, 2)
            z_mask = mask[:, 2]
            Rz = batch[properties.R][:, 2]
            idx = 0
            xt_z = torch.zeros_like(Rz)
            for i, n in enumerate(batch["_n_atoms"]):
                xt_z[idx : (idx + n)] = TruncatedNormal(
                    Rz[idx : (idx + n)],
                    stds[idx : (idx + n), 0],
                    z_confinement[i, 0],
                    z_confinement[i, 1],
                ).sample()
                idx += n

            xt_z[z_mask] = Rz[z_mask]

        xt = batch["_positions"].clone() + stds * noise

        if batch.get("z_confinement", None) is not None:
            xt[:, 2] = xt_z
            noise = (xt - batch["_positions"]) / stds

        noised_batch["_positions"] = xt

        score = self.forward(noised_batch, t, prob=self.train_prob)
        score[mask] = 0

        for i, cell in enumerate(batch["_cell"]):
            idx = batch["_idx_m"] == i
            noise[idx] = self.periodic_distance(xt[idx], noise[idx], cell)

        score_loss = torch.mean(torch.sum((noise + score * vars) ** 2, dim=-1))

        if self.potential_model is None:
            return {"loss": score_loss}

        ### energy/force loss
        targets = {
            "energy": batch[properties.energy],
            "forces": batch[properties.forces],
        }

        # batch = self.preprocess_batch(batch, save_keys=[])
        outputs = self.potential_model(batch)
        # energy
        pot_loss = self.loss_config["energy_weight"] * self.loss_config["energy_fn"](
            outputs["energy"], targets["energy"]
        )
        # forces
        pot_loss += self.loss_config["forces_weight"] * self.loss_config["forces_fn"](
            outputs["forces"], targets["forces"]
        )

        losses = {
            "loss": score_loss + pot_loss,
            "score_loss": score_loss,
            "pot_loss": pot_loss,
        }
        return losses

    @torch.no_grad()
    def sample(self, batch, num_steps=1000, save_traj=False, w=None, condition=None):
        """
        This implements the Euler-Maruyama method for sampling from a diffusion process.
        """
        batch = self.batch_random_positions(batch)
        if num_steps == 0 and not save_traj:
            return batch

        time_steps = torch.linspace(1, self.eps, num_steps)
        step_size = time_steps[0] - time_steps[1]
        mask = batch.get(
            "mask", torch.zeros_like(batch[properties.R], dtype=torch.bool)
        ).bool()

        if save_traj:
            traj = [self.batch_clone(batch)]

        for time in time_steps:
            t = torch.ones((batch["_idx_m"].shape[0], 1)) * time
            disp = self.dispersion(time)

            s = self.sample_forward(batch, t, w=w, condition=condition)

            drift = disp**2 * s - self.forward_drift(batch[properties.R], t)
            noise = torch.randn_like(batch[properties.R])

            if batch.get("z_confinement", None) is None:
                x_step = step_size * drift + torch.sqrt(step_size) * disp * noise
                x_step[mask] = 0

                batch[properties.R] = batch[properties.R] + x_step

            else:
                x_step = step_size * drift
                x_step[mask] = 0
                batch[properties.R] = batch[properties.R] + x_step

                noise_step = torch.sqrt(step_size) * disp * noise

                B = batch["_n_atoms"].shape[0]
                z_confinement = batch["z_confinement"].view(B, 2)
                Rz = batch[properties.R][:, 2]

                idx = 0
                xz = torch.zeros_like(Rz)
                for i, n in enumerate(batch["_n_atoms"]):
                    xz[idx : (idx + n)] = TruncatedNormal(
                        Rz[idx : (idx + n)],
                        torch.sqrt(step_size) * disp,
                        z_confinement[i, 0],
                        z_confinement[i, 1],
                    ).sample()
                    idx += n

                noise_step[:, 2] = xz - batch[properties.R][:, 2]
                noise_step[mask] = 0

                batch[properties.R] = batch[properties.R] + noise_step
                print(f"mean z at t={time}:", batch[properties.R][:, 2].mean())

            batch = self.batch_wrap(batch)

            if save_traj:
                clone = self.batch_clone(batch)
                clone["score"] = s
                traj.append(clone)

        if save_traj:
            return batch, traj
        else:
            return batch

    @torch.no_grad()
    def regressor_guidance_sample(
        self, batch, num_steps=1000, save_traj=False, w=None, condition=None, eta=1e-3
    ):
        """
        This implements the Euler-Maruyama method for sampling from a diffusion process.
        """
        assert self.potential_model is not None, "Potential model is not defined."

        time_steps = torch.linspace(1, 0, num_steps)
        step_size = time_steps[0] - time_steps[1]
        batch = self.batch_random_positions(batch)

        mask = batch.get(
            "mask", torch.zeros_like(batch[properties.R], dtype=torch.bool)
        ).bool()

        if save_traj:
            traj = [self.batch_clone(batch)]

        for time in time_steps:
            t = torch.ones((batch["_idx_m"].shape[0], 1)) * time
            disp = self.dispersion(time)

            batch = self.potential(batch)
            if time > self.eps:
                s = self.sample_forward(batch, t, w=w, condition=condition)
                drift = disp**2 * s - self.forward_drift(batch[properties.R], t)
                noise = torch.randn_like(batch[properties.R])
            else:
                drift = torch.zeros_like(batch[properties.R])
                noise = torch.zeros_like(batch[properties.R])

            E, F = batch[properties.energy], batch[properties.forces]
            guidance = (1 - time) * eta * F

            if batch.get("z_confinement", None) is None:
                x_step = (
                    step_size * drift + torch.sqrt(step_size) * disp * noise + guidance
                )
                x_step[mask] = 0
                batch[properties.R] = batch[properties.R] + x_step

            else:
                x_step = step_size * drift + guidance
                x_step[mask] = 0

                if not x_step.isfinite().all():
                    x_step[~x_step.isfinite()] = 0

                batch[properties.R] = batch[properties.R] + x_step

                noise_step = torch.sqrt(step_size) * disp * noise

                B = batch["_n_atoms"].shape[0]
                z_confinement = batch["z_confinement"].view(B, 2)
                Rz = batch[properties.R][:, 2]

                idx = 0
                xz = torch.zeros_like(Rz)
                for i, n in enumerate(batch["_n_atoms"]):
                    xz[idx : (idx + n)] = TruncatedNormal(
                        Rz[idx : (idx + n)],
                        torch.sqrt(step_size) * disp,
                        z_confinement[i, 0],
                        z_confinement[i, 1],
                    ).sample()
                    idx += n

                if not xz.isfinite().all():
                    xz[~xz.isfinite()] = batch[properties.R][:, 2][~xz.isfinite()]

                noise_step[:, 2] = xz - batch[properties.R][:, 2]
                noise_step[mask] = 0

                batch[properties.R] = batch[properties.R] + noise_step

            print(
                f"time: {time:.2f}, Mean energy: {E.mean():.2f}, Mean height: {batch[properties.R][:,2].mean():.2f}"
            )

            batch = self.batch_wrap(batch)

            if save_traj:
                clone = self.batch_clone(batch)
                clone["score"] = s
                traj.append(clone)

        # for time in time_steps:
        #     t = torch.ones((batch["_idx_m"].shape[0], 1)) * time
        #     disp = self.dispersion(time)

        #     batch = self.potential(batch)
        #     s = self.sample_forward(batch, t, w=w, condition=condition)

        #     E, F = batch[properties.energy], batch[properties.forces]
        #     drift = disp**2 * s - self.forward_drift(batch[properties.R], t)

        #     # # run idx 2
        #     # if time < 0.1:
        #     #     noise = t * torch.randn_like(batch[properties.R]) + (1-time) * F
        #     # else:
        #     #     noise = torch.randn_like(batch[properties.R])

        #     # # run idx 3
        #     guidance = (1-time) * 0.01 * F
        #     noise = torch.randn_like(batch[properties.R]) + guidance

        #     # # run idx 4: baseline
        #     # noise = torch.randn_like(batch[properties.R])

        #     x_step = step_size * drift + torch.sqrt(step_size) * disp * noise
        #     x_step[mask] = 0

        #     batch[properties.R] = batch[properties.R] + x_step

        #     print(
        #         f"time: {time:.2f}, Mean energy: {E.mean():.2f}, Mean height: {batch[properties.R][:,2].mean():.2f}"
        #     )

        # batch = self.batch_wrap(batch)

        # if save_traj:
        #     clone = self.batch_clone(batch)
        #     clone["score"] = s
        #     traj.append(clone)

        if save_traj:
            return batch, traj
        else:
            return batch
