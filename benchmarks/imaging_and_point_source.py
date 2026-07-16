"""
Benchmark: Joint Imaging + Point-Source Lensing (RXJ1131-1231)
==============================================================

Jointly models the extended arcs of RXJ1131-1231's lensed host galaxy (real
HST H-band imaging) and its point-source constraints (HST image positions +
COSMOGRAIL time delays) within the same lens model — the analysis pattern
behind time-delay cosmography.

**Paired workspace example:**
`autolens_workspace/scripts/multi/features/imaging_and_point_source/modeling.py`
fits the identical datasets with the identical model (see it for the
demonstration-grade noise-map/PSF caveats of the hips2fits imaging).

Run:

    python benchmarks/imaging_and_point_source.py            # full benchmark
    python benchmarks/imaging_and_point_source.py --quick    # fast smoke run
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

HIPS2FITS_URL = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    "?hips=CDS%2FP%2FHST%2FH&ra=172.96446&dec=-12.53293"
    "&width=200&height=200&fov=0.00333&projection=TAN&format=fits"
)

bench = harness.Benchmark(
    name="imaging_and_point_source",
    paired_example="scripts/multi/features/imaging_and_point_source/modeling.py",
    description="RXJ1131-1231 joint fit: HST H-band arcs + image positions + time "
    "delays, one SIE+shear mass model.",
    quick=args.quick,
)

"""
__Dataset__

Identical preparation to the paired example: hips2fits HST H-band cutout
(border-RMS noise-map, Gaussian FWHM 0.19" PSF), quasar images noise-scaled
out of the pixels, 3.0" mask; the point dataset is the same published
positions + time delays as `benchmarks/point_source.py`.
"""
with bench.phase("dataset"):
    pixel_scales = 0.06
    data_fits = harness.fetch_url(HIPS2FITS_URL, "multi/rxj1131", "data.fits")

    data = al.Array2D.from_fits(file_path=data_fits, pixel_scales=pixel_scales)
    data = al.Array2D.no_mask(
        values=np.nan_to_num(np.asarray(data.native)), pixel_scales=pixel_scales
    )

    data_np = np.asarray(data.native)
    border = np.concatenate(
        [data_np[:20, :].ravel(), data_np[-20:, :].ravel(),
         data_np[:, :20].ravel(), data_np[:, -20:].ravel()]
    )
    clipped = border[np.abs(border - np.median(border)) < 3.0 * np.std(border)]
    background_rms = float(np.std(clipped))

    noise_map = al.Array2D.full(
        fill_value=background_rms, shape_native=data.shape_native, pixel_scales=pixel_scales
    )
    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.19 / 2.355, pixel_scales=pixel_scales
    )

    dataset_imaging = al.Imaging(data=data, noise_map=noise_map, psf=psf)

    positions = al.Grid2DIrregular(
        [(-0.520, -2.037), (0.662, -2.076), (-1.632, -1.460), (0.356, 1.074)]
    )
    dataset_point = al.PointDataset(
        name="point_0",
        positions=positions,
        positions_noise_map=al.ArrayIrregular([0.005] * 4),
        time_delays=al.ArrayIrregular(values=[0.7, 0.0, -0.4, 91.4]),
        time_delays_noise_map=al.ArrayIrregular(values=[1.4, 1.4, 2.0, 1.5]),
    )

    grid_all = al.Grid2D.uniform(shape_native=data.shape_native, pixel_scales=pixel_scales)
    circles = np.zeros(data.shape_native, dtype=bool)
    for centre in np.asarray(dataset_point.positions):
        distances = np.hypot(
            np.asarray(grid_all.native)[:, :, 0] - centre[0],
            np.asarray(grid_all.native)[:, :, 1] - centre[1],
        )
        circles |= distances < 0.3
    mask_quasar = al.Mask2D(mask=np.invert(circles), pixel_scales=pixel_scales)
    dataset_imaging = dataset_imaging.apply_noise_scaling(mask=mask_quasar)

    mask_radius = 3.0
    mask = al.Mask2D.circular(
        shape_native=dataset_imaging.shape_native,
        pixel_scales=pixel_scales,
        radius=mask_radius,
    )
    dataset_imaging = dataset_imaging.apply_mask(mask=mask)

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset_imaging.grid,
        sub_size_list=[4, 2, 1],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )
    dataset_imaging = dataset_imaging.apply_over_sampling(over_sample_size_lp=over_sample_size)

"""
__Model__

Shared SIE + external shear mass model; the imaging view adds MGE lens light
and an MGE host-galaxy source, the point view the `Point` source constrained
by positions and delays.
"""
with bench.phase("model"):
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model_imaging = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.295, bulge=lens_bulge, mass=mass, shear=shear),
            source=af.Model(al.Galaxy, redshift=0.658, bulge=source_bulge),
        )
    )
    model_point = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.295, mass=mass, shear=shear),
            source=af.Model(al.Galaxy, redshift=0.658, point_0=af.Model(al.ps.Point)),
        )
    )

    solver = al.PointSolver.for_grid(
        grid=al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.2),
        pixel_scale_precision=0.001,
        magnification_threshold=0.1,
    )

    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(
            prior_model=model_imaging,
            analysis=al.AnalysisImaging(dataset=dataset_imaging, use_jax=True),
        ),
        af.AnalysisFactor(
            prior_model=model_point,
            analysis=al.AnalysisPoint(dataset=dataset_point, solver=solver, use_jax=True),
        ),
        use_jax=True,
    )

bench.measure_compile(
    model=factor_graph.global_prior_model,
    analysis=factor_graph,
    n_parallel=args.n_batch,
    use_grad=False,
)

search = af.Nautilus(
    path_prefix=Path("jax_joss"),
    name="imaging_and_point_source" + ("_quick" if args.quick else ""),
    unique_tag="rxj1131",
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
        "mass": "Isothermal + ExternalShear (shared across factors)",
        "imaging": "HST H-band arcs, MGE lens light + MGE source",
        "point": "4 positions + 3 relative time delays",
    },
)
