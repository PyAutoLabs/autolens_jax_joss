"""
Benchmark: Joint Strong + Weak Lensing (Abell 2744)
===================================================

Constrains a single cluster-scale mass model of Abell 2744 using both its
strong-lensing multiple images (Bergamini et al. 2023 gold systems) and its
weak-lensing shear catalogue (pyRRG / Harvey & Massey 2024) — real data on
both sides, in one factor-graph likelihood.

**Paired workspace example:**
`autolens_workspace/scripts/weak/features/strong_lensing/a2744.py` fits the
identical datasets with the identical model; both datasets share one
coordinate frame (arc-second offsets about the projected cluster core).

Run:

    python benchmarks/strong_and_weak.py            # full benchmark
    python benchmarks/strong_and_weak.py --quick    # fast smoke run
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

CATALOGUE_URL = (
    "https://raw.githubusercontent.com/davidharvey1986/pyRRG/"
    "0ccc29fb4513137da61b1afb632ca492093bd609/"
    "trainStarGalClass/TrainingData/abell2744_galaxies.fits"
)

bench = harness.Benchmark(
    name="strong_and_weak",
    paired_example="scripts/weak/features/strong_lensing/a2744.py",
    description="Abell 2744 joint fit: 25 multiple images (7 multi-plane sources) + "
    "~400 weak-lensing shear measurements constraining one cluster mass model.",
    quick=args.quick,
)

"""
__Dataset__

Strong: the committed cluster CSVs of the paired example. Weak: the pyRRG
shape catalogue with the standard cuts, projected about the same centre.
"""
with bench.phase("dataset"):
    dataset_path = harness.fetch_workspace_dataset(
        "cluster/a2744",
        ["point_datasets.csv", "scaling_galaxies.csv", "mass.csv", "point.csv"],
    )
    dataset_list = al.list_from_csv(file_path=dataset_path / "point_datasets.csv")
    mass_table = al.galaxy_models_from_csv(file_path=dataset_path / "mass.csv", family="mass")
    point_table = al.galaxy_models_from_csv(file_path=dataset_path / "point.csv", family="point")
    scaling_galaxies_table = al.galaxy_table_from_csv(
        file_path=dataset_path / "scaling_galaxies.csv"
    )

    catalogue_path = harness.fetch_url(
        CATALOGUE_URL, "weak/a2744_pyrrg", "abell2744_galaxies.fits"
    )

    from astropy.io import fits as astropy_fits

    with astropy_fits.open(catalogue_path) as hdul:
        table = hdul[1].data

    ra = np.asarray(table["ra"], dtype=float)
    dec = np.asarray(table["dec"], dtype=float)
    e1 = np.asarray(table["e1"], dtype=float)
    e2 = np.asarray(table["e2"], dtype=float)
    e1_err = np.asarray(table["e1_err"], dtype=float)
    e2_err = np.asarray(table["e2_err"], dtype=float)

    ra_centre, dec_centre = 3.5875, -30.3972
    x = (ra - ra_centre) * np.cos(np.deg2rad(dec_centre)) * 3600.0
    y = (dec - dec_centre) * 3600.0
    radii = np.sqrt(x**2.0 + y**2.0)

    use = (
        np.isfinite(e1) & np.isfinite(e2) & np.isfinite(e1_err) & np.isfinite(e2_err)
        & (np.abs(e1) < 1.0) & (np.abs(e2) < 1.0)
        & (e1_err > 0.0) & (e1_err < 0.4) & (e2_err > 0.0) & (e2_err < 0.4)
        & (radii > 10.0) & (radii < 130.0)
    )

    sigma_int = 0.25
    noise = np.sqrt(sigma_int**2.0 + 0.5 * (e1_err[use] ** 2.0 + e2_err[use] ** 2.0))

    dataset_weak = al.WeakDataset.from_arrays(
        positions=np.stack([y[use], x[use]], axis=1),
        gamma_1=e1[use],
        gamma_2=e2[use],
        noise_map=list(noise),
        is_reduced=True,
        name="a2744_pyrrg",
    )

"""
__Model__

The four-tier cluster model of the paired example; the strong factors receive
the full multi-plane view, the weak factor the same mass-model objects with a
single effective z = 1.0 source plane — shared model objects means shared
priors, the definition of a joint fit.
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
        point_attr.centre_0 = af.GaussianPrior(mean=float(np.mean(positions[:, 0])), sigma=3.0)
        point_attr.centre_1 = af.GaussianPrior(mean=float(np.mean(positions[:, 1])), sigma=3.0)

    scaling_b0_ref = af.UniformPrior(lower_limit=0.0, upper_limit=1.0)
    scaling_galaxies_list = []
    for centre, luminosity in zip(
        scaling_galaxies_table.centres, scaling_galaxies_table.luminosities
    ):
        ratio = luminosity / 1.0
        mass = af.Model(al.mp.dPIEMassSph)
        mass.centre = tuple(centre)
        mass.ra = 0.158 * ratio**0.5
        mass.rs = 15.8 * ratio**0.5
        mass.b0 = scaling_b0_ref * ratio**0.5
        scaling_galaxies_list.append(af.Model(al.Galaxy, redshift=redshift_lens, mass=mass))

    scaling_galaxies = af.Collection(scaling_galaxies_list)

    model_strong = af.Collection(
        galaxies=af.Collection(**galaxy_models),
        scaling_galaxies=scaling_galaxies,
    )

    model_weak = af.Collection(
        galaxies=af.Collection(
            lens_0=galaxy_models["lens_0"],
            lens_1=galaxy_models["lens_1"],
            host_halo=galaxy_models["host_halo"],
            source_weak=af.Model(al.Galaxy, redshift=1.0),
        ),
        scaling_galaxies=scaling_galaxies,
    )

    grid = al.Grid2D.uniform(shape_native=(120, 120), pixel_scales=1.0)
    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )

    analysis_factor_list = [
        af.AnalysisFactor(
            prior_model=model_strong,
            analysis=al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=True),
        )
        for dataset in dataset_list
    ]
    analysis_factor_list.append(
        af.AnalysisFactor(
            prior_model=model_weak,
            analysis=al.AnalysisWeak(dataset=dataset_weak, use_jax=True),
        )
    )

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

bench.measure_compile(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    n_parallel=args.n_batch,
    use_grad=False,
)

search = af.Nautilus(
    path_prefix=Path("jax_joss"),
    name="strong_and_weak" + ("_quick" if args.quick else ""),
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
        "strong": f"{len(dataset_list)} multi-plane point sources (25 images)",
        "weak": "~400 reduced-shear measurements (effective z=1.0 plane)",
        "mass_model": "2 dPIE BCGs + NFW host halo + 188-member scaling tier (shared)",
    },
)
