"""
Benchmark: Galaxy-Scale CCD Imaging (JWST COSMOS-Web Ring F150W)
================================================================

Models JWST imaging of the COSMOS-Web Ring — a spectacular Einstein ring
discovered in COSMOS-Web (Mercier et al. 2024) — with lens-light subtraction
and a pixelized reconstruction of the lensed source, using gradient-based
inference with JAX automatic differentiation.

**Paired workspace example:** `autolens_workspace/scripts/imaging/start_here.py`
uses the identical dataset (fetched from the workspace repository on first
run, so the data is byte-identical). The workspace example teaches the
workflow with an MGE source; this benchmark uses the paper's benchmark model:

- Lens light: 20-Gaussian Multi-Gaussian Expansion (MGE).
- Lens mass: Singular Isothermal Ellipsoid (SIE) + external shear.
- Source: pixelized reconstruction on a rectangular kernel-CDF mesh
  (`RectangularKernelAdaptDensity`), whose likelihood is certified
  differentiable at `over_sample_size_pixelization=1`.

Run:

    python benchmarks/imaging.py            # full benchmark (GPU recommended)
    python benchmarks/imaging.py --quick    # minutes-long smoke run (any machine)
    python benchmarks/imaging.py --search nautilus   # gradient-free baseline
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
parser.add_argument("--search", choices=["adam", "nautilus"], default="adam")
parser.add_argument("--n-starts", type=int, default=16, help="multi-start Adam broad starts")
parser.add_argument("--n-steps", type=int, default=300, help="Adam steps per start")
parser.add_argument(
    "--batch-size",
    type=int,
    default=4,
    help="lax.map batch size bounding memory; the batched gradient of a pixelized "
    "likelihood at 16 parallel starts exhausts even an 80 GB A100 unbatched",
)
parser.add_argument("--mesh-pixels", type=int, default=30, help="source mesh pixels per side")
args = parser.parse_args()

mask_radius = 2.5

if args.quick:
    args.n_starts, args.n_steps, args.mesh_pixels = 2, 5, 10
    args.batch_size = 1
    mask_radius = 1.6

bench = harness.Benchmark(
    name="imaging" + ("_nautilus" if args.search == "nautilus" else ""),
    paired_example="scripts/imaging/start_here.py",
    description="Galaxy-scale JWST COSMOS-Web Ring F150W imaging: MGE lens light "
    "subtraction + SIE+shear mass + pixelized source reconstruction.",
    quick=args.quick,
)

"""
__Dataset__

Identical loading to the paired start_here.py: 0.06" pixels, extra-galaxies
noise scaling, a 2.5" circular mask (1.6" in --quick mode) and radially-binned
over sampling.
"""
with bench.phase("dataset"):
    dataset_path = harness.fetch_workspace_dataset(
        "imaging/cosmos_web_ring",
        ["data.fits", "psf.fits", "noise_map.fits", "mask_extra_galaxies.fits"],
    )

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=0.06,
    )

    mask_extra_galaxies = al.Mask2D.from_fits(
        file_path=dataset_path / "mask_extra_galaxies.fits",
        pixel_scales=dataset.pixel_scales,
        invert=True,
    )
    dataset = dataset.apply_noise_scaling(mask=mask_extra_galaxies)

    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=mask_radius,
    )
    dataset = dataset.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset = dataset.apply_over_sampling(
        over_sample_size_lp=over_sample_size,
        over_sample_size_pixelization=1,
    )

"""
__Model__

The paper's benchmark model: MGE lens light + SIE+shear mass + a pixelized
source on the kernel-CDF rectangular mesh (differentiable at pixelization
over-sampling of 1).
"""
with bench.phase("model"):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    mesh = af.Model(
        al.mesh.RectangularKernelAdaptDensity,
        shape=(args.mesh_pixels, args.mesh_pixels),
        bandwidth=0.1,
    )
    regularization = af.Model(al.reg.Constant)
    pixelization = af.Model(al.Pixelization, mesh=mesh, regularization=regularization)
    source = af.Model(al.Galaxy, redshift=1.0, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Positions Likelihood__

Pixelized-source fits require a positions-likelihood penalty to exclude
demagnified solutions. The positions below are the brightest arc pixel in each
quadrant of a 0.5"-1.4" annulus around the ring, extracted from the data
itself; they only need to trace the lensed source approximately.
"""
positions = al.Grid2DIrregular(
    [(0.36, 0.54), (0.54, -0.06), (-0.36, 0.72), (-0.36, -0.78)]
)
positions_likelihood = al.PositionsLH(positions=positions, threshold=0.3)

analysis = al.AnalysisImaging(
    dataset=dataset,
    positions_likelihood_list=[positions_likelihood],
    use_jax=True,
)

"""
__Compile Measurement__

Compile the same jit(vmap(value_and_grad(likelihood))) program the search
driver builds and time first-call vs warm-call — this is the paper's "JAX
compilation time".
"""
bench.measure_compile(
    model=model,
    analysis=analysis,
    n_parallel=args.n_starts,
    map_batch_size=args.batch_size,
)

"""
__Search__

Multi-start Adam: broad parallel starts with the best kept, the robust-and-fast
recipe for JAX lens likelihoods (single cold starts land in the wrong basin).
`--search nautilus` runs the gradient-free nested-sampling baseline instead.
"""
if args.search == "adam":
    search = af.MultiStartAdam(
        path_prefix=Path("jax_joss"),
        name="imaging" + ("_quick" if args.quick else ""),
        unique_tag="cosmos_web_ring",
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        iterations_per_quick_update=10**9,
    )
    n_evals = args.n_starts * args.n_steps
else:
    search = af.Nautilus(
        path_prefix=Path("jax_joss"),
        name="imaging_nautilus" + ("_quick" if args.quick else ""),
        unique_tag="cosmos_web_ring",
        n_live=50 if args.quick else 100,
        n_batch=50,
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
        "batch_size": args.batch_size,
    },
    model_info={
        "lens": "MGE(20) bulge + Isothermal + ExternalShear",
        "source": f"RectangularKernelAdaptDensity({args.mesh_pixels}x{args.mesh_pixels}, "
        "bandwidth=0.1) + Constant regularization",
        "mask_radius_arcsec": mask_radius,
        "image_pixels_in_mask": int(dataset.grid.shape[0]),
    },
)
