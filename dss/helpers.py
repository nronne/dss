def get_dataset(
    atoms,
    neighbour_list,
    path="dataset.db",
):
    import os

    import numpy as np
    import schnetpack.transform as trn
    from ase.calculators.singlepoint import SinglePointCalculator
    from schnetpack.data import ASEAtomsData, AtomsDataModule

    print("=" * 10, "Creating dataset", "=" * 10)
    if os.path.exists(path):
        os.remove(path)
        if os.path.exists("split.npz"):
            os.remove("split.npz")

    property_list = []
    for a in atoms:
        e = a.get_potential_energy()
        f = a.get_forces().reshape(-1, 3)
        c = SinglePointCalculator(a, energy=e, forces=f)
        a.set_calculator(c)
        mask = np.zeros_like(f, dtype=bool)
        mask[a.get_positions()[:, 2] < 2.5] = True

        z_confinement = [2.5, 8]

        properties = {
            "energy": np.array([e]),
            "forces": f,
            "mask": mask,
            "z_confinement": z_confinement,
        }
        property_list.append(properties)

    dataset = ASEAtomsData.create(
        path,
        distance_unit="Ang",
        property_unit_dict={
            "energy": "eV",
            "forces": "eV/Ang",
            "mask": None,
            "z_confinement": None,
        },
    )
    dataset.add_systems(property_list, atoms)

    print("Number of reference calculations:", len(dataset))
    print("Available properties:")

    for p in dataset.available_properties:
        print("-", p)

    example = dataset[0]
    print("Properties of item in dataset:")

    for k, v in example.items():
        print("-", k, ":", v.shape)

    print("=" * 10, "Finished dataset", "=" * 10)

    dataset = AtomsDataModule(
        path,
        batch_size=32,
        num_train=0.90,
        num_val=0.1,
        transforms=[
            neighbour_list,
            trn.CastTo32(),
        ],
        num_workers=32,
        pin_memory=True,
        split_file="split.npz",
    )
    dataset.prepare_data()
    dataset.setup()

    train_dataloader = dataset.train_dataloader()
    example = next(iter(train_dataloader))

    print("Properties of batch:")

    for k, v in example.items():
        print("-", k, ":", v.shape)

    print("idx:", example["_idx_m"])

    return dataset

def get_diffusion_model(cutoff=6.0):
    import schnetpack as spk

    from dss.diffusion import VPDiffusion
    from dss.models import (ConditionalPaiNN, ConditionedScoreModel,
                            NodeEmbedding, Potential)
    from dss.utils import TorchNeighborList

    neighbour_list = TorchNeighborList(cutoff=cutoff)

    n_atom_basis = 64
    radial_basis = spk.nn.GaussianRBF(n_rbf=30, cutoff=cutoff)
    embedding = NodeEmbedding(n_atom_basis, time_dim=0)
    representation = ConditionalPaiNN(
        embedding=embedding,
        n_interactions=4,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
    )

    score_model = ConditionedScoreModel(representation, time_dim=2, gated_blocks=4)

    pred_energy = spk.atomistic.Atomwise(
        n_in=representation.embedding.dim, output_key="energy"
    )
    pred_forces = spk.atomistic.Forces(energy_key="energy", force_key="forces")
    pairwise_distance = spk.atomistic.PairwiseDistances()
    potential = Potential(
        representation=representation,
        input_modules=[pairwise_distance],
        output_modules=[pred_energy, pred_forces],
        postprocessors=[
            # trn.CastTo64(),
            # trn.AddOffsets("energy", add_atomrefs=False, add_mean=True),
        ],
    )

    diffusion = VPDiffusion(
        score_model=score_model,
        potential_model=potential,
        neighbour_list=neighbour_list,
        beta_max=3.0,
        optim_config={"lr": 1e-3},
        scheduler_config={"factor": 0.90, "patience": 100},
    )

    return diffusion, neighbour_list

def sample(diffusion, num_samples):
    import numpy as np
    import schnetpack as spk
    import torch
    from ase import Atoms
    from ase.io import read, write

    def to_atoms(batch_list):
        atoms = []
        for b in batch_list:
            a = Atoms(
                numbers=b["_atomic_numbers"].detach().numpy(),
                positions=b["_positions"].detach().numpy(),
                cell=b["_cell"].detach().numpy().reshape(3, 3),
                pbc=b["_pbc"].detach().numpy(),
            )
            atoms.append(a)
        return atoms

    N_Ag, N_O = 6, 3
    template = read("/home/roenne/documents/pgm/examples/AgxOy/ag111-3x3.traj")
    # template = read('/home/roenne/documents/pgm/examples/AgxOy/template.traj')

    converter = spk.interfaces.AtomsConverter(
        neighbor_list=None,
        additional_inputs={
            "mask": torch.tensor(
                np.vstack(
                    [
                        np.ones((len(template), 3), dtype=bool),
                        np.zeros((N_O + N_Ag, 3), dtype=bool),
                    ]
                )
            ),
            "z_confinement": torch.tensor(np.array([2.5, 8.0])),
        },
    )

    # generate data
    n_split = 50

    all_atoms = []
    for i in range(num_samples // n_split):
        atoms_data = []
        for _ in range(n_split):
            symbols = ["Ag"] * len(template) + ["Ag"] * N_Ag + ["O"] * N_O
            positions = np.vstack(
                (template.get_positions(), np.zeros((len(symbols) - len(template), 3)))
            )
            atoms_data.append(
                Atoms(
                    symbols,
                    positions=positions,
                    cell=template.get_cell(),
                    pbc=template.get_pbc(),
                )
            )

        data = converter(atoms_data)
        data["_pbc"] = data["_pbc"].view(-1)  # hack

        batch = diffusion.regressor_guidance_sample(
            data, num_steps=1000, save_traj=False, eta=1e-2
        )

        # #save traj
        # all_trajs = [[] for _ in range(n_split)]
        # for b in traj:
        #     batch_list = diffusion._split_batch(b)
        #     for j, item in enumerate(batch_list):
        #         all_trajs[j].append(item)

        # for j, batch_list in enumerate(all_trajs):
        #     atoms = to_atoms(batch_list)
        #     write(f'trajs-zconfined/sampled_{i*n_split + j}.traj', atoms)

        # save final
        batch_list = diffusion._split_batch(batch)
        atoms = to_atoms(batch_list)
        all_atoms += atoms

        write("Ag6O3_Ag9_sampled_confined_RG.traj", all_atoms)