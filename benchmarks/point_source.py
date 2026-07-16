"""
Benchmark: Point-Source Lensing (RXJ1131-1231, positions + time delays)
=======================================================================

Models the quadruply imaged quasar RXJ1131-1231 using real point-source
observables: the four HST-measured image positions (Suyu et al. 2013, ApJ 766,
70, Table 1; 0.005" precision, lens-galaxy-centred (y,x) arcsec) and the
COSMOGRAIL time delays (Tewes et al. 2013, A&A 556, A22; relative to image B:
0.7 +/- 1.4, -0.4 +/- 2.0 and 91.4 +/- 1.5 days for A, C, D).

**Paired workspace example:** `autolens_workspace/scripts/point_source/start_here.py`
fits the identical dataset — these values are committed there as
`dataset/point_source/rxj1131/point_dataset_with_time_delays.json`.

The likelihood ray-traces triangles through the mass model to solve for the
multiple images of the point source, entirely in JAX; batches of parameter
vectors evaluate in parallel on the GPU. The fit uses nested sampling on that
GPU-batched likelihood: gradients through the iterative multi-image solve are
not part of the certified differentiable surface, and the likelihood is cheap
enough that sampling comfortably meets the time budget.

Run:

    python benchmarks/point_source.py            # full benchmark
    python benchmarks/point_source.py --quick    # fast smoke run
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autofit as af
import autolens as al

import harness

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true", help="fast smoke run, results not official")
parser.add_argument("--n-live", type=int, default=75)
parser.add_argument("--n-batch", type=int, default=50)
args = parser.parse_args()

if args.quick:
    args.n_live, args.n_batch = 30, 10

bench = harness.Benchmark(
    name="point_source",
    paired_example="scripts/point_source/start_here.py",
    description="RXJ1131-1231 quad quasar: image positions (Suyu+13) + COSMOGRAIL "
    "time delays (Tewes+13), SIE+shear mass model.",
    quick=args.quick,
)

"""
__Dataset__

The same published values committed to the workspace as
`dataset/point_source/rxj1131/point_dataset_with_time_delays.json`.
Images ordered A, B, C, D; delays relative to image B (the reference image
carries the smallest measured pairwise uncertainty).
"""
with bench.phase("dataset"):
    positions = al.Grid2DIrregular(
        [
            (-0.520, -2.037),  # Image A
            (0.662, -2.076),  # Image B
            (-1.632, -1.460),  # Image C
            (0.356, 1.074),  # Image D
        ]
    )

    dataset = al.PointDataset(
        name="point_0",
        positions=positions,
        positions_noise_map=al.ArrayIrregular([0.005] * 4),
        time_delays=al.ArrayIrregular(values=[0.7, 0.0, -0.4, 91.4]),
        time_delays_noise_map=al.ArrayIrregular(values=[1.4, 1.4, 2.0, 1.5]),
    )

"""
__Model__

SIE + external shear lens mass at the measured redshifts (z_lens = 0.295,
z_source = 0.658 — the distances that convert Fermat potential differences
into delays in days), with the point source's (y,x) source-plane centre free.
"""
with bench.phase("model"):
    grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2)

    solver = al.PointSolver.for_grid(
        grid=grid, pixel_scale_precision=0.001, magnification_threshold=0.1
    )

    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    lens = af.Model(al.Galaxy, redshift=0.295, mass=mass, shear=shear)

    point_0 = af.Model(al.ps.Point)
    source = af.Model(al.Galaxy, redshift=0.658, point_0=point_0)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisPoint(dataset=dataset, solver=solver, use_jax=True)

bench.measure_compile(
    model=model, analysis=analysis, n_parallel=args.n_batch, use_grad=False
)

"""
__Search__

Nested sampling (Nautilus) over the GPU-batched JAX point likelihood.
"""
search = af.Nautilus(
    path_prefix=Path("jax_joss"),
    name="point_source" + ("_quick" if args.quick else ""),
    unique_tag="rxj1131",
    n_live=args.n_live,
    n_batch=args.n_batch,
    iterations_per_quick_update=10**9,
)

result = bench.run_search(search=search, model=model, analysis=analysis)

bench.finish(
    search_info={"type": "Nautilus", "n_live": args.n_live, "n_batch": args.n_batch},
    model_info={
        "lens": "Isothermal + ExternalShear (z=0.295)",
        "source": "Point (z=0.658)",
        "observables": "4 image positions + 3 relative time delays",
    },
)
