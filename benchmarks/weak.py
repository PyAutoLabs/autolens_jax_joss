"""
Benchmark: Weak Lensing (Abell 2744 JWST shape catalogue)
=========================================================

Fits a real weak-lensing shear catalogue — the Abell 2744 galaxy shape
catalogue shipped with the public pyRRG shape-measurement code (Harvey &
Massey 2024, MNRAS 529, 802) — with a differentiable JAX likelihood,
demonstrating that PyAutoLens-JAX is not restricted to strong-lensing data.

**Paired workspace example:** `autolens_workspace/scripts/weak/start_here.py`
uses the identical catalogue (same pinned pyRRG commit), the identical quality
cuts, and the identical spherical-NFW halo model.

Run:

    python benchmarks/weak.py            # full benchmark
    python benchmarks/weak.py --quick    # fast smoke run
    python benchmarks/weak.py --search nautilus
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
parser.add_argument("--search", choices=["adam", "nautilus"], default="adam")
parser.add_argument("--n-starts", type=int, default=16)
parser.add_argument("--n-steps", type=int, default=300)
args = parser.parse_args()

if args.quick:
    args.n_starts, args.n_steps = 2, 5

CATALOGUE_URL = (
    "https://raw.githubusercontent.com/davidharvey1986/pyRRG/"
    "0ccc29fb4513137da61b1afb632ca492093bd609/"
    "trainStarGalClass/TrainingData/abell2744_galaxies.fits"
)

bench = harness.Benchmark(
    name="weak",
    paired_example="scripts/weak/start_here.py",
    description="Abell 2744 weak-lensing shape catalogue (pyRRG / Harvey & Massey "
    "2024): spherical NFW halo fitted to reduced-shear measurements.",
    quick=args.quick,
)

"""
__Dataset__

Identical to the paired start_here.py: tangent-plane projection about the
cluster core, the standard quality cuts, intrinsic shape dispersion of 0.25
per component combined in quadrature with measurement noise, reduced shear.
"""
with bench.phase("dataset"):
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

    ra_centre, dec_centre = 3.5875, -30.3972  # A2744 core (J2000 degrees)

    x = (ra - ra_centre) * np.cos(np.deg2rad(dec_centre)) * 3600.0
    y = (dec - dec_centre) * 3600.0
    radii = np.sqrt(x**2.0 + y**2.0)

    finite = (
        np.isfinite(e1) & np.isfinite(e2) & np.isfinite(e1_err) & np.isfinite(e2_err)
    )
    physical = (np.abs(e1) < 1.0) & (np.abs(e2) < 1.0)
    well_measured = (e1_err > 0.0) & (e1_err < 0.4) & (e2_err > 0.0) & (e2_err < 0.4)
    radial = (radii > 10.0) & (radii < 130.0)
    use = finite & physical & well_measured & radial

    sigma_int = 0.25
    noise = np.sqrt(sigma_int**2.0 + 0.5 * (e1_err[use] ** 2.0 + e2_err[use] ** 2.0))

    dataset = al.WeakDataset.from_arrays(
        positions=np.stack([y[use], x[use]], axis=1),
        gamma_1=e1[use],
        gamma_2=e2[use],
        noise_map=list(noise),
        is_reduced=True,
        name="a2744_pyrrg",
    )

"""
__Model__

Spherical NFW halo at z = 0.308 with a single effective source plane at
z = 1.0, centre with 10" Gaussian priors about the projected cluster core —
identical to the paired example.
"""
with bench.phase("model"):
    mass = af.Model(al.mp.NFWSph)
    mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=10.0)
    mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=10.0)

    lens = af.Model(al.Galaxy, redshift=0.308, mass=mass)
    source = af.Model(al.Galaxy, redshift=1.0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisWeak(dataset=dataset, use_jax=True)

bench.measure_compile(model=model, analysis=analysis, n_parallel=args.n_starts)

"""
__Search__
"""
if args.search == "adam":
    search = af.MultiStartAdam(
        path_prefix=Path("jax_joss"),
        name="weak" + ("_quick" if args.quick else ""),
        unique_tag="a2744_pyrrg",
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        iterations_per_quick_update=10**9,
    )
    n_evals = args.n_starts * args.n_steps
else:
    search = af.Nautilus(
        path_prefix=Path("jax_joss"),
        name="weak_nautilus" + ("_quick" if args.quick else ""),
        unique_tag="a2744_pyrrg",
        n_live=50 if args.quick else 100,
        iterations_per_quick_update=10**9,
    )
    n_evals = None

result = bench.run_search(
    search=search, model=model, analysis=analysis, n_likelihood_evals=n_evals
)

bench.finish(
    search_info={
        "type": type(search).__name__,
        "n_starts": args.n_starts if args.search == "adam" else None,
        "n_steps": args.n_steps if args.search == "adam" else None,
    },
    model_info={
        "lens": "NFWSph halo (z=0.308)",
        "source_plane": "single effective plane at z=1.0",
        "n_galaxies": int(dataset.shear_yx.shape[0]) if hasattr(dataset, "shear_yx") else None,
    },
)
