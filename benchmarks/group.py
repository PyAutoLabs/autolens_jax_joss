"""
Benchmark: Group-Scale Strong Lensing (real Euclid group lens, HST imaging)
===========================================================================

Models a real group-scale strong lens — a Euclid-discovered system with
multiple deflecting galaxies, observed with HST — demonstrating that
PyAutoLens-JAX is not restricted to isolated galaxy-scale lenses. Each main
lens galaxy carries its own MGE light and isothermal mass profile; the source
is an MGE.

**Paired workspace example:** `autolens_workspace/scripts/group/start_here.py`
uses the identical dataset (fetched from the workspace repository on first
run) and the identical model family.

Run:

    python benchmarks/group.py            # full benchmark (GPU recommended)
    python benchmarks/group.py --quick    # fast smoke run
    python benchmarks/group.py --search nautilus
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
parser.add_argument("--n-starts", type=int, default=16)
parser.add_argument("--n-steps", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=None)
args = parser.parse_args()

if args.quick:
    args.n_starts, args.n_steps = 2, 5
    args.batch_size = 1

DATASET = "102021990_NEG650312660474055399"

bench = harness.Benchmark(
    name="group",
    paired_example="scripts/group/start_here.py",
    description="Real Euclid group-scale lens (HST imaging): per-galaxy MGE light + "
    "isothermal mass for each main lens galaxy, MGE source.",
    quick=args.quick,
)

"""
__Dataset__

Identical loading to the paired start_here.py: 0.1" pixels, a 3.7" circular
mask enclosing all main lens galaxies, over sampling centred on each.
"""
with bench.phase("dataset"):
    dataset_path = harness.fetch_workspace_dataset(
        f"group/{DATASET}",
        ["data.fits", "psf.fits", "noise_map.fits", "main_lens_centres.json"],
    )

    dataset = al.Imaging.from_fits(
        data_path=dataset_path / "data.fits",
        psf_path=dataset_path / "psf.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        pixel_scales=0.1,
    )

    main_lens_centres = al.from_json(file_path=dataset_path / "main_lens_centres.json")

    mask_radius = 3.7
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
        centre_list=list(main_lens_centres),
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

Identical to the paired example: one MGE bulge + Isothermal mass per main lens
galaxy (shear on the first only), MGE source.
"""
with bench.phase("model"):
    lens_dict = {}
    for i, centre in enumerate(main_lens_centres):
        bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            centre_prior_is_uniform=True,
            centre=(centre[0], centre[1]),
        )
        mass = af.Model(al.mp.Isothermal)
        mass.centre = (centre[0], centre[1])

        lens_dict[f"lens_{i}"] = af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=bulge,
            mass=mass,
            shear=af.Model(al.mp.ExternalShear) if i == 0 else None,
        )

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius,
        total_gaussians=20,
        gaussian_per_basis=1,
        centre_prior_is_uniform=False,
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    model = af.Collection(galaxies=af.Collection(**lens_dict, source=source))

analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

bench.measure_compile(
    model=model,
    analysis=analysis,
    n_parallel=args.n_starts,
    map_batch_size=args.batch_size,
)

"""
__Search__
"""
if args.search == "adam":
    search = af.MultiStartAdam(
        path_prefix=Path("jax_joss"),
        name="group" + ("_quick" if args.quick else ""),
        unique_tag=DATASET,
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        iterations_per_quick_update=10**9,
    )
    n_evals = args.n_starts * args.n_steps
else:
    search = af.Nautilus(
        path_prefix=Path("jax_joss"),
        name="group_nautilus" + ("_quick" if args.quick else ""),
        unique_tag=DATASET,
        n_live=50 if args.quick else 150,
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
        "main_lens_galaxies": len(main_lens_centres),
        "lens": "per-galaxy MGE(20) + Isothermal (+ shear on lens_0)",
        "source": "MGE(20) bulge",
        "mask_radius_arcsec": mask_radius,
    },
)
