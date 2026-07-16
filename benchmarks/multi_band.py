"""
Benchmark: Multi-Band Imaging (JWST COSMOS-Web Ring, 4 bands)
=============================================================

Jointly models all four available JWST bands of the COSMOS-Web Ring
(F115W, F150W, F277W, F444W), constraining a common lens mass model while
fitting the wavelength-dependent lens and source emission in each dataset —
the paper's demonstration that multiple datasets combine into a single
differentiable, GPU-accelerated probabilistic model.

**Paired workspace example:** `autolens_workspace/scripts/multi/start_here.py`
uses the identical waveband datasets (fetched from the workspace repository on
first run) and the identical model family: per-band MGE lens/source emission,
a shared SIE+shear mass model, and free sub-pixel dataset offsets. (The
workspace example enables two bands by default to run fast for new users;
this benchmark fits all four, as the paper specifies.)

Run:

    python benchmarks/multi_band.py            # full benchmark (GPU recommended)
    python benchmarks/multi_band.py --quick    # minutes-long smoke run
    python benchmarks/multi_band.py --search nautilus
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
parser.add_argument("--quick", action="store_true", help="fast smoke run: 2 bands, few steps")
parser.add_argument("--search", choices=["adam", "nautilus"], default="adam")
parser.add_argument("--n-starts", type=int, default=16)
parser.add_argument("--n-steps", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=None)
args = parser.parse_args()

waveband_list = ["F115W", "F150W", "F277W", "F444W"]
pixel_scale_dict = {"F115W": 0.03, "F150W": 0.03, "F277W": 0.06, "F444W": 0.06}

if args.quick:
    args.n_starts, args.n_steps = 2, 5
    args.batch_size = 1
    waveband_list = ["F277W", "F444W"]

bench = harness.Benchmark(
    name="multi_band",
    paired_example="scripts/multi/start_here.py",
    description=f"Joint modeling of {len(waveband_list)} JWST COSMOS-Web Ring bands: "
    "common SIE+shear mass, per-band MGE lens/source emission, free dataset offsets.",
    quick=args.quick,
)

"""
__Dataset__

Identical loading to the paired multi/start_here.py: per-band pixel scales,
extra-galaxies noise scaling, 2.5" mask and radially-binned over sampling.
"""
mask_radius = 2.5
dataset_list = []

with bench.phase("dataset"):
    for waveband in waveband_list:
        dataset_path = harness.fetch_workspace_dataset(
            f"imaging/cosmos_web_ring/wavebands/{waveband}",
            ["data.fits", "psf.fits", "noise_map.fits", "mask_extra_galaxies.fits"],
        )

        dataset = al.Imaging.from_fits(
            data_path=dataset_path / "data.fits",
            psf_path=dataset_path / "psf.fits",
            noise_map_path=dataset_path / "noise_map.fits",
            pixel_scales=pixel_scale_dict[waveband],
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
        dataset = dataset.apply_over_sampling(over_sample_size_lp=over_sample_size)

        dataset_list.append(dataset)

"""
__Model__

Identical to the paired example: MGE lens light + SIE+shear + MGE source, plus
a `DatasetModel` whose sub-pixel grid offsets are freed for every band after
the first.
"""
with bench.phase("model"):
    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    lens = af.Model(al.Galaxy, redshift=0.5, bulge=bulge, mass=mass, shear=shear)

    bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )
    source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

    dataset_model = af.Model(al.DatasetModel)

    model = af.Collection(
        dataset_model=dataset_model, galaxies=af.Collection(lens=lens, source=source)
    )

    analysis_factor_list = []
    for i, dataset in enumerate(dataset_list):
        analysis = al.AnalysisImaging(dataset=dataset, use_jax=True)

        model_analysis = model.copy()
        if i > 0:
            model_analysis.dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )
            model_analysis.dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
                lower_limit=-1.0, upper_limit=1.0
            )

        analysis_factor_list.append(
            af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)
        )

    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

bench.measure_compile(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    n_parallel=args.n_starts,
    map_batch_size=args.batch_size,
)

"""
__Search__
"""
if args.search == "adam":
    search = af.MultiStartAdam(
        path_prefix=Path("jax_joss"),
        name="multi_band" + ("_quick" if args.quick else ""),
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
        name="multi_band_nautilus" + ("_quick" if args.quick else ""),
        unique_tag="cosmos_web_ring",
        n_live=50 if args.quick else 150,
        n_batch=50,
        iterations_per_quick_update=10**9,
    )
    n_evals = None

result = bench.run_search(
    search=search,
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    n_likelihood_evals=n_evals,
)

bench.finish(
    search_info={
        "type": type(search).__name__,
        "n_starts": args.n_starts if args.search == "adam" else None,
        "n_steps": args.n_steps if args.search == "adam" else None,
        "batch_size": args.batch_size,
    },
    model_info={
        "wavebands": waveband_list,
        "lens": "MGE(20) bulge + Isothermal + ExternalShear (shared across bands)",
        "source": "MGE(20) bulge per band",
        "dataset_offsets": "free sub-pixel offsets for bands 2+",
        "mask_radius_arcsec": mask_radius,
    },
)
