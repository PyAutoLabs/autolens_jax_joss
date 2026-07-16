"""
Benchmark: Joint Imaging + Interferometry (SDP.81)
==================================================

Jointly fits HST H-band imaging of SDP.81 (the lens galaxy and the
near-infrared emission of the lensed source) and its ALMA long-baseline
visibilities (the submillimetre dust emission of the same source),
constraining a common SIE+shear mass model with complementary observations of
the lensed galaxy.

**Paired workspace example:**
`autolens_workspace/scripts/multi/features/imaging_and_interferometer/`
teaches the same joint-fit pattern.

**Data:** the imaging half is fetched automatically (CDS hips2fits HST H-band
cutout, demonstration-grade noise/PSF as in the RXJ1131 benchmarks); the ALMA
visibilities require the one-off CASA export of `data_prep/sdp81/README.md`.

Run:

    python benchmarks/imaging_and_interferometer.py            # full benchmark
    python benchmarks/imaging_and_interferometer.py --quick    # fast smoke run
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

SDP81_URL = None  # see benchmarks/interferometer.py
FILES = ["data.fits", "noise_map.fits", "uv_wavelengths.fits"]

HIPS2FITS_URL = (
    "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    "?hips=CDS%2FP%2FHST%2FH&ra=135.79876&dec=0.65225"
    "&width=200&height=200&fov=0.00333&projection=TAN&format=fits"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true", help="fast smoke run, results not official")
parser.add_argument("--n-live", type=int, default=150)
parser.add_argument("--n-batch", type=int, default=20)
parser.add_argument("--mesh-pixels", type=int, default=30)
args = parser.parse_args()

if args.quick:
    args.n_live, args.n_batch, args.mesh_pixels = 30, 5, 10

uv_path = harness.DATASET_DIR / "interferometer" / "sdp81"
if not all((uv_path / f).exists() for f in FILES):
    if SDP81_URL is not None:
        for f in FILES:
            harness.fetch_url(f"{SDP81_URL}/{f}", "interferometer/sdp81", f)
    else:
        sys.exit(
            "SDP.81 visibility FITS files not found — see data_prep/sdp81/README.md."
        )

bench = harness.Benchmark(
    name="imaging_and_interferometer",
    paired_example="scripts/multi/features/imaging_and_interferometer/modeling.py",
    description="SDP.81 joint fit: HST H-band imaging + ALMA visibilities, one "
    "SIE+shear mass model, per-dataset source reconstructions.",
    quick=args.quick,
)

"""
__Dataset__
"""
with bench.phase("dataset"):
    pixel_scales = 0.06
    data_fits = harness.fetch_url(HIPS2FITS_URL, "multi/sdp81", "data.fits")

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
    noise_map = al.Array2D.full(
        fill_value=float(np.std(clipped)),
        shape_native=data.shape_native,
        pixel_scales=pixel_scales,
    )
    psf = al.Convolver.from_gaussian(
        shape_native=(11, 11), sigma=0.19 / 2.355, pixel_scales=pixel_scales
    )
    dataset_imaging = al.Imaging(data=data, noise_map=noise_map, psf=psf)

    mask = al.Mask2D.circular(
        shape_native=dataset_imaging.shape_native, pixel_scales=pixel_scales, radius=3.0
    )
    dataset_imaging = dataset_imaging.apply_mask(mask=mask)

    real_space_mask = al.Mask2D.circular(
        shape_native=(200, 200), pixel_scales=0.035, radius=3.5
    )
    dataset_interferometer = al.Interferometer.from_fits(
        data_path=uv_path / "data.fits",
        noise_map_path=uv_path / "noise_map.fits",
        uv_wavelengths_path=uv_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
    )

"""
__Model__

Shared SIE + shear mass (z_lens = 0.299); the imaging view carries MGE lens
light and an MGE near-infrared source, the interferometer view a pixelized
submillimetre source (z_source = 3.042) — the two source reconstructions are
independent, as the emission arises from different physical components.
"""
with bench.phase("model"):
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)

    mask_radius = 3.0
    lens_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=True
    )
    source_bulge = al.model_util.mge_model_from(
        mask_radius=mask_radius, total_gaussians=20, centre_prior_is_uniform=False
    )

    model_imaging = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.299, bulge=lens_bulge, mass=mass, shear=shear),
            source=af.Model(al.Galaxy, redshift=3.042, bulge=source_bulge),
        )
    )

    mesh = af.Model(
        al.mesh.RectangularKernelAdaptDensity,
        shape=(args.mesh_pixels, args.mesh_pixels),
        bandwidth=0.1,
    )
    pixelization = af.Model(
        al.Pixelization, mesh=mesh, regularization=af.Model(al.reg.Constant)
    )
    model_interferometer = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(al.Galaxy, redshift=0.299, mass=mass, shear=shear),
            source=af.Model(al.Galaxy, redshift=3.042, pixelization=pixelization),
        )
    )

    factor_graph = af.FactorGraphModel(
        af.AnalysisFactor(
            prior_model=model_imaging,
            analysis=al.AnalysisImaging(dataset=dataset_imaging, use_jax=True),
        ),
        af.AnalysisFactor(
            prior_model=model_interferometer,
            analysis=al.AnalysisInterferometer(dataset=dataset_interferometer, use_jax=True),
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
    name="imaging_and_interferometer" + ("_quick" if args.quick else ""),
    unique_tag="sdp81",
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
        "mass": "Isothermal + ExternalShear (shared, z=0.299)",
        "imaging": "HST H-band, MGE lens light + MGE NIR source",
        "interferometer": f"ALMA visibilities, pixelized submm source "
        f"({args.mesh_pixels}x{args.mesh_pixels})",
    },
)
