from typing import Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from schnetpack import properties  # change to use this at some point
from schnetpack.atomistic import PairwiseDistances
from schnetpack.data.loader import _atoms_collate_fn
from schnetpack.transform import WrapPositions

from dss.utils import OFFSET_LIST, TruncatedNormal


class VPDiffusion(pl.LightningModule):

    """
    Implements variance-preserving diffusion (VP-Diffusion) using a score model.

    """

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
        verbose=True,
    ):
        """
        Args
        ______
        score_model: torch.nn.Module
            A score model that takes a batch of data and outputs a score for each atom.
        neighbour_list: schnetpack.atomistic.NeighbourList
            A neighbour list that takes a batch of data and outputs a neighbour list
            for each atom.
        potential_model: torch.nn.Module
            A potential model that takes a batch of data and outputs the energy and
            forces for each atom.
        pairwise_distance: schnetpack.atomistic.PairwiseDistances
            A pairwise distance module that takes a batch of data and outputs
            the pairwise distances for each atom.
        potential_head: bool
            Whether to use the potential model as a head of the score model.
        beta_min: float
            The minimum value of beta.
        beta_max: float
            The maximum value of beta.
        eps: float
            A small value to add to the variance to avoid numerical issues.
        condition_config: dict
            A dictionary of configuration options for conditioning the score model.
        loss_config: dict
            A dictionary of configuration options for the loss function.
        optim_config: dict
            A dictionary of configuration options for the optimizer.
        scheduler_config: dict
            A dictionary of configuration options for the learning rate scheduler.

        """
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

        if self.score_model.cond_dim > 0:
            self.train_prob = condition_config.get("train_prob", 0.5)
        else:
            self.train_prob = 0.0

        self.condition_config = condition_config
        self.optim_config = optim_config
        self.scheduler_config = scheduler_config
        self.verbose = verbose

    ### Diffusion Methods ###

    def beta_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the value of beta at time t.

        Args
        ______
        t: torch.Tensor
            The time at which to calculate beta.

        Returns
        _______
        beta: torch.Tensor
            The value of beta at time t.

        """

        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the value of alpha at time t.

        Args
        ______
        t: torch.Tensor
            The time at which to calculate alpha.

        Returns
        _______
        alpha: torch.Tensor
            The value of alpha at time t.

        """
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def forward_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the drift term of the diffusion process.

        Args
        ______
        x: torch.Tensor
            The positions of the atoms.
        t: torch.Tensor
            The time at which to calculate the drift term.

        Returns
        _______
        drift: torch.Tensor
            The drift term of the diffusion process.
        """
        return -0.5 * self.beta_t(t) * x

    def dispersion(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the dispersion term of the diffusion process.

        Args
        ______
        t: torch.Tensor
            The time at which to calculate the dispersion term.

        Returns
        _______
        dispersion: torch.Tensor
            The dispersion term of the diffusion process.
        """
        return torch.sqrt(self.beta_t(t))

    def mean_factor(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mean factor of the diffusion process.

        Args
        ______
        t: torch.Tensor
            The time at which to calculate the mean factor.

        Returns
        _______
        mean_factor: torch.Tensor
            The mean factor of the diffusion process.
        """
        return torch.exp(-self.alpha_t(t))

    def marginal_probability(self, t: torch.Tensor) -> torch.Tensor:
        """
        Calculates the marginal probability of the diffusion process.

        Args
        ______
        t: torch.Tensor
            The time at which to calculate the marginal probability.

        Returns
        _______
        marginal_probability: torch.Tensor
            The marginal probability of the diffusion process.
        """
        return 1 - torch.exp(-self.alpha_t(t))

    ### LightningModule Methods ###
    def setup(self, stage: str = None) -> None:
        """
        Sets up the model.

        Args
        ______
        stage: str
            The stage of training.

        """

        self.offsets = torch.tensor(OFFSET_LIST).float().to(self.device)

    def training_step(self, batch: Dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Performs a training step.

        Args
        ______
        batch: dict
            A batch of data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        _______
        loss: torch.Tensor
            The loss of the training step.

        """
        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            self.log("train_" + k, v)
        return losses["loss"]

    def validation_step(self, batch: Dict, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Performs a validation step.

        Args
        ______
        batch: dict
            A batch of data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        _______
        loss: torch.Tensor
            The loss of the validation step.

        """
        if self.potential_model is not None:
            torch.set_grad_enabled(True)

        losses = self.loss(batch, batch_idx)
        for k, v in losses.items():
            self.log("val_" + k, v)
        return losses["loss"]

    def configure_optimizers(self) -> Dict:
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        _______
        optimizers: dict
            A dictionary of optimizers and learning rate schedulers.


        """
        optimizer = torch.optim.Adam(self.parameters(), **self.optim_config)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_config
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def loss(self, batch: Dict, batch_idx: torch.Tensor) -> Dict:
        """
        Calculates the loss.

        Args
        ______
        batch: dict
            A batch of data.
        batch_idx: torch.Tensor
            The index of the batch.

        Returns
        _______
        losses: dict
            A dictionary of losses.

        """
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

    def forward(
        self,
        batch: Dict,
        t: torch.Tensor,
        prob: float = 0.0,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs a forward pass.

        Args
        ______
        batch: dict
            A batch of data.
        t: torch.Tensor
            The random variable.
        prob: float
            The probability of using the conditional model.
        condition: torch.Tensor
            The condition for the conditional model.

        Returns
        _______
        score: torch.Tensor
            The score.

        """
        if (
            "scalar_representation" not in batch
            and "vector_representation" not in batch
        ):
            batch = self.preprocess_batch(batch)

        v = self.score_model(batch, t, prob=prob, condition=condition).view(
            batch[properties.R].shape
        )
        return v

    ### Methods ###

    def sample_forward(
        self,
        batch: Dict,
        t: torch.Tensor,
        w: float = None,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs a forward sampling

        Args
        ______
        batch: dict
            A batch of data.
        t: torch.Tensor
            The random variable.
        w: float
            The weight of the conditional model.
        condition: torch.Tensor
            The condition for the conditional model.

        Returns
        _______
        score: torch.Tensor
            The score.ing

        """
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
    def potential(self, batch: Dict):
        """
        Predict energy and forces

        Args
        ______
        batch: dict
            A batch of data.

        Returns
        _______
        batch: dict
            The batch of data with the predicted energy and forces

        """
        batch = self.preprocess_batch(batch, save_keys=[])
        self.potential_model(batch)
        return batch

    def periodic_distance(
        self, X: torch.Tensor, N: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        """
        Takes X and N (noise) and computes the minimum distance between X and Y=X+N
        taking into account periodic boundary conditions.

        Args
        ______
        X: torch.Tensor
            The positions (N, 3)
        N: torch.Tensor
            The noise (N, 3)
        cell: torch.Tensor
            The cell (3, 3)

        Returns
        _______
        dist: torch.Tensor
            The distance between X and Y=X+N
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

    def preprocess_batch(
        self, batch: Dict, save_keys: List = ["energy", "forces"]
    ) -> Dict:
        """
        Preprocess the batch of data by calculating neighbourlist and pairwise
        distances between atoms

        Args
        ______
        batch: dict
            A batch of data.
        save_keys: list
            The keys to save in the batch

        Returns
        _______
        batch: dict
            The batch of data with the calculated neighbourlist and pairwise
            distances between atoms

        """

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
            
        batch = {p: batch[p].to(self.device) for p in batch}
        return batch

    ### Samplers ###

    @torch.no_grad()
    def sample(
        self,
        batch: Dict,
        num_steps: int = 1000,
        save_traj: bool = False,
        w: float = None,
        condition: torch.Tensor = None,
    ) -> Dict:
        """
        This implements the Euler-Maruyama method for sampling from a diffusion process.

        Args
        ______
        batch: dict
            A batch of data.
        num_steps: int
            The number of steps to sample.
        save_traj: bool
            Whether to save the trajectory.
        w: float
            The weight of the conditional model.
        condition: torch.Tensor
            The condition for the conditional model.

        Returns
        _______
        batch: dict
            The batch of data with the sampled trajectory.
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
            t = torch.ones((batch["_idx_m"].shape[0], 1), device=self.device) * time
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
        self,
        batch: Dict,
        num_steps: int=1000,
        save_traj: bool=False,
        w: float=None,
        condition: torch.Tensor=None,
        eta: float=1e-3,
        postrelax_steps=100,
        fmax=0.05,) -> Dict:
        """
        This implements the Euler-Maruyama method for sampling from a diffusion process
        using regressor-guidance.

        Args
        ______
        batch: dict
            A batch of data.
        num_steps: int
            The number of steps to sample.
        save_traj: bool
            Whether to save the trajectory.
        w: float
            The weight of the conditional model.
        condition: torch.Tensor
            The condition for the conditional model.
        eta: float
            The learning rate for the regressor.
        
        """
        assert self.potential_model is not None, "Potential model is not defined."

        batch = self.batch_random_positions(batch)
        if num_steps == 0:
            return batch

        time_steps = torch.linspace(1, self.eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        
        mask = batch.get(
            "mask", torch.zeros_like(batch[properties.R], dtype=torch.bool)
        ).bool()

        if save_traj:
            traj = [self.batch_clone(batch)]

        for time in time_steps:
            t = torch.ones((batch["_idx_m"].shape[0], 1), device=self.device) * time
            disp = self.dispersion(time)

            batch = self.potential(batch)
            s = self.sample_forward(batch, t, w=w, condition=condition)
            drift = disp**2 * s - self.forward_drift(batch[properties.R], t)
            noise = torch.randn_like(batch[properties.R])

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

                old_R = batch[properties.R].clone()
                batch[properties.R] = batch[properties.R] + x_step

                noise_step = torch.sqrt(step_size) * disp * noise

                B = batch["_n_atoms"].shape[0]
                z_confinement = batch["z_confinement"].view(B, 2)
                Rz = batch[properties.R][:, 2]

                idx = 0
                xz = torch.zeros_like(Rz)
                n_out = torch.zeros_like(batch["_n_atoms"])
                for i, n in enumerate(batch["_n_atoms"]):
                    # check if any atoms are moved outside z_confinement by score model
                    out_idx = (Rz[idx : (idx + n)] < z_confinement[i,0]) + (Rz[idx : (idx + n)] > z_confinement[i,1])
                    out_idx[mask[idx : (idx + n), 2]] = False
                    
                    if out_idx.any():
                        n_out[i] = 1
                        # Now move them inside again at random
                        random_xz = torch.rand_like(Rz[idx : (idx + n)])*(z_confinement[i, 1]-z_confinement[i, 0])+z_confinement[i, 0]
                        Rz[idx : (idx + n)][out_idx] = random_xz[out_idx]

                        
                        Rxy = batch[properties.R][idx : (idx + n), :2].clone()
                        cell = batch[properties.cell][i, :, :].clone().view(3, 3)[:2,:2]
                        fxy = torch.rand_like(Rxy)
                        Rxy[out_idx] = torch.matmul(fxy[out_idx], cell)
                        batch[properties.R][idx : (idx + n), :2][out_idx] = Rxy[out_idx] # old_R[idx : (idx + n), :2][out_idx] # 
                        

                    xz[idx : (idx + n)] = TruncatedNormal(
                        Rz[idx : (idx + n)],
                        torch.sqrt(step_size) * disp,
                        z_confinement[i, 0],
                        z_confinement[i, 1],
                    ).sample()

                    # old except
                    #     print('failed z confinement truncation.')
                    #     new_xz = Rz[idx : (idx + n)].clone()
                    #     fail_idx = (new_xz < z_confinement[i, 0]) + (new_xz > z_confinement[i, 1])
                    #     random_xz = torch.rand_like(new_xz)*(z_confinement[i, 1]-z_confinement[i, 0])+z_confinement[i, 0]
                    #     new_xz[fail_idx] = random_xz[fail_idx]
                    #     # new_xz[mask[idx:(idx+n),2]] = Rz[idx : (idx + n)][mask[idx:(idx+n),2]]
                        
                    #     xz[idx : (idx + n)] = new_xz
                        
                    idx += n

                if not xz.isfinite().all():
                    xz[~xz.isfinite()] = batch[properties.R][:, 2][~xz.isfinite()]

                noise_step[:, 2] = xz - batch[properties.R][:, 2]
                noise_step[mask] = 0

                batch[properties.R] = batch[properties.R] + noise_step


            if self.verbose:
                print(
                    f"time: {time:.2f}, Mean energy: {E.mean():.2f}, Mean height: {batch[properties.R][:,2].mean():.2f}, Moved outside confinement: {sum(n_out)}/{len(n_out)}"
                )

            batch = self.batch_wrap(batch)
            # self.batch_check_fusion(batch)
            
            if save_traj:
                clone = self.batch_clone(batch)
                clone["score"] = s
                traj.append(clone)

        batch = self.potential(batch)

        if eta > 0:
            for _ in range(postrelax_steps):
                F = batch[properties.forces]
                F = torch.nan_to_num(F, nan=0.0)                
                if (F < fmax).all():
                    break
                
                step = eta * F
                step[mask] = 0
                batch[properties.R] = batch[properties.R] + step
                
                batch = self.potential(batch)

        if save_traj:
            return batch, traj
        else:
            return batch

    ### Utility Methods ###

    # Make decorator for batching functions like this
    def batch_wrap(self, batch: Dict) -> Dict:  
        """
        Wraps all atoms into periodic cell in batch.

        Args
        ______
        batch: dict
            A batch of data.

        Returns
        _______
        batch: dict
            A batch of data with wrapped atoms.
        
        """
        batch_list = self._split_batch(batch)
        for i in range(len(batch_list)):
            batch_list[i] = self.wrap_positions(batch_list[i])

        batch = self._collate_batch(batch_list)
        return batch

    def batch_check_fusion(self, batch: Dict) -> Dict:  
        """
        Wraps all atoms into periodic cell in batch.

        Args
        ______
        batch: dict
            A batch of data.

        Returns
        _______
        batch: dict
            A batch of data with wrapped atoms.
        
        """
        batch_list = self._split_batch(batch)
        for i in range(len(batch_list)):
            batch_list[i] = self._check_fusion(batch_list[i])

        batch = self._collate_batch(batch_list)
        return batch

    def batch_clone(self, batch: Dict, ignore: List=()) -> Dict:
        """
        Clones a batch of data.

        Args
        ______
        batch: dict
            A batch of data.
        ignore: list
            A list of properties to ignore.

        Returns
        _______
        batch: dict
            A cloned batch of data.
        
        """
        batch_copy = {}
        for key in batch.keys():
            if key not in ignore:
                batch_copy[key] = batch[key].clone()

        return batch_copy

    def batch_random_positions(self, batch: Dict) -> Dict:
        """
        Randomizes the positions of atoms in a batch.

        Args
        ______
        batch: dict
            A batch of data.

        Returns
        _______
        batch: dict
            A batch of data with randomized positions.
        
        """
        batch_list = self._split_batch(batch)
        for i in range(len(batch_list)):
            batch_list[i] = self._random_positions(batch_list[i])

        batch = self._collate_batch(batch_list)
        return batch

    def _split_batch(self, batch: Dict, keep_ef=False) -> List:
        """
        Splits a batch of data into a list of batches.

        Args
        ______
        batch: dict
            A batch of data.
        keep_ef: bool
            Whether to keep energy and forces.

        Returns
        _______
        batch_list: list
            A list of batches.
        
        """
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

        if keep_ef:
            energy = batch.get("energy", None)
            forces = batch.get("forces", None)
        else:
            energy, forces = None, None

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

            if energy is not None:
                inputs["energy"] = energy[idx]
            if forces is not None:
                inputs["forces"] = forces[idx_c : idx_c + curr_n_atoms]
                

            idx_c += curr_n_atoms
            output_list.append(inputs)

        return output_list

    def _collate_batch(self, batch_list: List) -> Dict:
        """
        Collates a list of batches into a single batch.

        Args
        ______
        batch_list: list
            A list of batches.

        Returns
        _______
        batch: dict
            A batch of data.
        
        """
        
        return _atoms_collate_fn(batch_list)

    def _random_positions(self, structure: Dict) -> Dict:
        """
        Randomizes the positions of atoms in a structure.

        Args
        ______
        structure: dict
            A structure.

        Returns
        _______
        structure: dict
            A structure with randomized positions.
        
        """
        
        positions = structure[properties.R]
        cell = structure[properties.cell].clone().view(3, 3)
        corner = torch.zeros(3, device=positions.device)

        if structure.get("z_confinement", None) is not None:
            cell[2, 2] = structure["z_confinement"][1] - structure["z_confinement"][0]
            corner[2] = structure["z_confinement"][0]

        mask = structure.get(
            "mask", torch.zeros_like(positions, dtype=torch.bool)
        ).bool()

        f = torch.rand_like(positions, device=positions.device)
        random_positions = torch.matmul(f, cell) + corner
        structure[properties.R][~mask] = random_positions[~mask]

        return structure
    
    def _check_fusion(self, structure: Dict) -> Dict:
        R = structure[properties.R]
        mask = structure.get(
            "mask", torch.zeros_like(structure[properties.R], dtype=torch.bool)
        ).bool()
        X = R[~mask].view(-1, 3)
        D = torch.cdist(X, X).fill_diagonal_(1e6)
        if D.min() < 1.:
            print("Atom fusion")
