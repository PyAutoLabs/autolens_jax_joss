"""
Benchmark: Interferometry (ALMA SDP.81, >1M visibilities)
=========================================================

Models the ALMA long-baseline Science Verification observations of SDP.81 —
more than one million interferometer visibilities of the z = 3.042 Einstein
ring — directly in the visibility domain with a pixelized source
reconstruction.

**Paired workspace example:** `autolens_workspace/scripts/interferometer/start_here.py`
(its SDP.81 upgrade tracks with this benchmark — see
autolens_workspace#281).

**Data:** the calibrated measurement sets are public but require a one-off
CASA export producing three averaging levels of the same dataset — see
`data_prep/sdp81/README.md`. Averaging keeps the long baselines, so every
level is genuinely ~25-30 mas resolution (real-space mask at 0.025"/pixel);
only the visibility count changes. `--nvis {default,mid,full}` selects the
level; running all three documents that per-evaluation runtime is ~flat in
N_vis. Place the exported folders under `dataset/interferometer/` or set
`SDP81_URL` to a public deposit.

Run:

    python benchmarks/interferometer.py            # full benchmark
    python benchmarks/interferometer.py --quick    # fast smoke run
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import autofit as af
import autolens as al

import harness

# Set to a public deposit (e.g. Zenodo) hosting the CASA-exported FITS files to
# enable automatic download; see data_prep/sdp81/README.md.
SDP81_URL = None

FILES = ["data.fits", "noise_map.fits", "uv_wavelengths.fits"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--quick", action="store_true", help="fast smoke run, results not official")
parser.add_argument("--search", choices=["adam", "nautilus"], default="adam")
parser.add_argument("--n-starts", type=int, default=16)
parser.add_argument("--n-steps", type=int, default=300)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--mesh-pixels", type=int, default=30)
parser.add_argument(
    "--nvis",
    choices=["default", "mid", "full"],
    default="default",
    help="averaging level: ~50k (default), ~500k (mid), >1M (full) visibilities",
)
args = parser.parse_args()

if args.quick:
    args.n_starts, args.n_steps, args.mesh_pixels = 2, 5, 10
    args.batch_size = 1

DATASET_FOLDER = {"default": "sdp81", "mid": "sdp81_mid", "full": "sdp81_full"}[args.nvis]

dataset_path = harness.DATASET_DIR / "interferometer" / DATASET_FOLDER

if not all((dataset_path / f).exists() for f in FILES):
    if SDP81_URL is not None:
        for f in FILES:
            harness.fetch_url(f"{SDP81_URL}/{DATASET_FOLDER}/{f}", f"interferometer/{DATASET_FOLDER}", f)
    else:
        sys.exit(
            "SDP.81 visibility FITS files not found in dataset/interferometer/sdp81/.\n"
            "They require a one-off CASA export from the public ALMA Science Verification\n"
            "measurement sets — see data_prep/sdp81/README.md for the two-step recipe\n"
            "(or set SDP81_URL in this script to a deposit hosting the exported files)."
        )

bench = harness.Benchmark(
    name="interferometer" + ("" if args.nvis == "default" else f"_{args.nvis}"),
    paired_example="scripts/interferometer/start_here.py",
    description="ALMA SDP.81 long-baseline visibilities (>1M): SIE+shear mass + "
    "pixelized source reconstruction in the uv-plane.",
    quick=args.quick,
)

"""
__Dataset__

A 3.5" real-space mask at 0.025"/pixel — the resolution the long-baseline
data genuinely warrants at every averaging level — defines the region the
source reconstruction maps; the likelihood is computed in the visibility
domain against all visibilities.
"""
with bench.phase("dataset"):
    real_space_mask = al.Mask2D.circular(
        shape_native=(300, 300), pixel_scales=0.025, radius=3.5
    )

    dataset = al.Interferometer.from_fits(
        data_path=dataset_path / "data.fits",
        noise_map_path=dataset_path / "noise_map.fits",
        uv_wavelengths_path=dataset_path / "uv_wavelengths.fits",
        real_space_mask=real_space_mask,
    )

"""
__Model__

SDP.81's lens light is negligible at ALMA wavelengths, so the model is
SIE + shear mass with a pixelized source on the kernel-CDF rectangular mesh
(differentiable), at the measured redshifts z_lens = 0.299, z_source = 3.042.
"""
with bench.phase("model"):
    mass = af.Model(al.mp.Isothermal)
    shear = af.Model(al.mp.ExternalShear)
    lens = af.Model(al.Galaxy, redshift=0.299, mass=mass, shear=shear)

    mesh = af.Model(
        al.mesh.RectangularKernelAdaptDensity,
        shape=(args.mesh_pixels, args.mesh_pixels),
        bandwidth=0.1,
    )
    pixelization = af.Model(
        al.Pixelization, mesh=mesh, regularization=af.Model(al.reg.Constant)
    )
    source = af.Model(al.Galaxy, redshift=3.042, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

analysis = al.AnalysisInterferometer(dataset=dataset, use_jax=True)

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
        name="interferometer" + ("_quick" if args.quick else ""),
        unique_tag=DATASET_FOLDER,
        n_starts=args.n_starts,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        iterations_per_quick_update=10**9,
    )
    n_evals = args.n_starts * args.n_steps
else:
    search = af.Nautilus(
        path_prefix=Path("jax_joss"),
        name="interferometer_nautilus" + ("_quick" if args.quick else ""),
        unique_tag=DATASET_FOLDER,
        n_live=50 if args.quick else 100,
        n_batch=20,
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
        "visibilities": int(dataset.data.shape[0]),
        "lens": "Isothermal + ExternalShear (z=0.299)",
        "source": f"RectangularKernelAdaptDensity({args.mesh_pixels}x{args.mesh_pixels}) (z=3.042)",
    },
)
