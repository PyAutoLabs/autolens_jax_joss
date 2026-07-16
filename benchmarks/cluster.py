"""
Benchmark: Cluster-Scale Strong Lensing (Abell 2744)
====================================================

Models the Hubble Frontier Fields cluster Abell 2744 ("Pandora's Cluster",
z = 0.308) using the published lens-model inputs of Bergamini et al. 2023
(A&A 670, A60): 7 gold multiple-image systems (25 images, spectroscopic
sources from z = 1.69 to z = 5.66 — genuinely multi-plane), two
individually-modelled BCGs, 188 scaling-relation cluster members and an NFW
host halo — multiple mass components, multiple images, multiple source
planes, as the paper specifies.

**Paired workspace example:** `autolens_workspace/scripts/cluster/start_here.py`
fits the identical dataset (`dataset/cluster/a2744/`, fetched from the
workspace repository on first run) with the identical model.

The likelihood ray-traces triangles through the multi-plane mass model to
solve every source's multiple images in JAX; nested sampling runs over the
GPU-batched likelihood (gradients through the iterative multi-image solve are
not part of the certified differentiable surface).

Run:

    python benchmarks/cluster.py            # full benchmark
    python benchmarks/cluster.py --quick    # fast smoke run
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import autofit as af
import autolens as al

import harness

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true", help="fast smoke run, results not official")
parser.add_argument("--n-live", type=int, default=150)
parser.add_argument("--n-batch", type=int, default=50)
args = parser.parse_args()

if args.quick:
    args.n_live, args.n_batch = 30, 10

bench = harness.Benchmark(
    name="cluster",
    paired_example="scripts/cluster/start_here.py",
    description="Abell 2744 (Bergamini+23 inputs): 7 gold multi-plane source systems, "
    "2 dPIE BCGs + 188 scaling members + NFW host halo.",
    quick=args.quick,
)

"""
__Dataset__

The committed CSVs of the paired example (full provenance in the workspace's
`dataset/cluster/a2744/README.md`).
"""
with bench.phase("dataset"):
    dataset_path = harness.fetch_workspace_dataset(
        "cluster/a2744",
        ["point_datasets.csv", "scaling_galaxies.csv", "mass.csv", "point.csv"],
    )

    dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")

    mass_table = al.galaxy_models_from_csv(
        file_path=dataset_path / "mass.csv", family="mass"
    )
    point_table = al.galaxy_models_from_csv(
        file_path=dataset_path / "point.csv", family="point"
    )

    scaling_galaxies_table = al.galaxy_table_from_csv(
        file_path=dataset_path / "scaling_galaxies.csv"
    )

"""
__Model__

Identical composition to the paired start_here.py: free dPIE parameters on the
two BCGs, free host-halo mass, free source centres, and one shared scaling
normalisation for all 188 members (22 free parameters in total).
"""
with bench.phase("model"):
    redshift_lens = 0.308

    galaxy_models = al.galaxy_af_models_from_csv_tables(mass_table, point_table)

    for name in ("lens_0", "lens_1"):
        galaxy_models[name].mass.ra = af.UniformPrior(lower_limit=1.0, upper_limit=15.0)
        galaxy_models[name].mass.rs = af.UniformPrior(lower_limit=5.0, upper_limit=40.0)
        galaxy_models[name].mass.b0 = af.UniformPrior(lower_limit=0.1, upper_limit=10.0)

    galaxy_models["host_halo"].dark.mass_at_200 = af.LogUniformPrior(
        lower_limit=10**14.5, upper_limit=10**16.0
    )

    for i, dataset in enumerate(dataset_list):
        positions = np.atleast_2d(dataset.positions)
        point_attr = getattr(galaxy_models[f"source_{i}"], f"point_{i}")
        point_attr.centre_0 = af.GaussianPrior(
            mean=float(np.mean(positions[:, 0])), sigma=3.0
        )
        point_attr.centre_1 = af.GaussianPrior(
            mean=float(np.mean(positions[:, 1])), sigma=3.0
        )

    scaling_b0_ref = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    scaling_exponent = 0.5
    reference_luminosity = 1.0  # member luminosities are normalised to the BCG
    scaling_ra_ref_fixed = 0.158
    scaling_rs_ref_fixed = 15.8

    scaling_galaxies_list = []
    for centre, luminosity in zip(
        scaling_galaxies_table.centres, scaling_galaxies_table.luminosities
    ):
        luminosity_ratio = luminosity / reference_luminosity

        mass = af.Model(al.mp.dPIEMassSph)
        mass.centre = tuple(centre)
        mass.ra = scaling_ra_ref_fixed * luminosity_ratio**scaling_exponent
        mass.rs = scaling_rs_ref_fixed * luminosity_ratio**scaling_exponent
        mass.b0 = scaling_b0_ref * luminosity_ratio**scaling_exponent

        scaling_galaxies_list.append(
            af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)
        )

    model = af.Collection(
        galaxies=af.Collection(**galaxy_models),
        scaling_galaxies=af.Collection(scaling_galaxies_list),
    )

    grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=1.0)
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )

    analysis_list = [
        al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=True)
        for dataset in dataset_list
    ]
    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model, analysis=analysis)
        for analysis in analysis_list
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

bench.measure_compile(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    n_parallel=args.n_batch,
    use_grad=False,
)

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=Path("jax_joss"),
    name="cluster" + ("_quick" if args.quick else ""),
    unique_tag="a2744",
    n_live=args.n_live,
    n_batch=args.n_batch,
    iterations_per_quick_update=10**9,
)

result = bench.run_search(
    search=search, model=factor_graph.global_prior_model, analysis=factor_graph
)

bench.finish(
    search_info={"type": "Nautilus", "n_live": args.n_live, "n_batch": args.n_batch},
    model_info={
        "sources": len(dataset_list),
        "images": int(sum(len(np.atleast_2d(d.positions)) for d in dataset_list)),
        "scaling_members": len(scaling_galaxies_table.luminosities),
        "tiers": "2 dPIE BCGs + NFW host halo + scaling tier + 7 Point sources",
        "source_redshifts": sorted(set(float(d.redshift) for d in dataset_list)),
    },
)
