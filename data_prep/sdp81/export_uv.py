"""
CASA export script: SDP.81 calibrated measurement set -> PyAutoLens FITS, at
three averaging levels (see README.md).

Run inside CASA (`casa -c export_uv.py <ms_path>`) or with pip-installed
casatools/casatasks (`python export_uv.py <ms_path>`). Writes
`sdp81/`, `sdp81_mid/` and `sdp81_full/` folders alongside this script, each
containing data.fits (real/imag), noise_map.fits (per-visibility RMS from the
MS weights) and uv_wavelengths.fits (u,v in wavelengths).
"""

import os
import shutil
import sys

import numpy as np

# (folder, timebin, chanbin) — averaging keeps the long baselines, so every
# level is genuinely long-baseline resolution; only N_vis changes.
LEVELS = [
    ("sdp81", "60s", 0),  # ~50k vis — the user-example / default benchmark level
    ("sdp81_mid", "10s", 4),  # ~500k vis
    ("sdp81_full", "2s", 0),  # >1M vis — the paper's headline row
]


def export_level(ms_path, out_dir, timebin, chanbin):
    from casatools import ms as ms_tool
    from casatasks import mstransform

    averaged = ms_path.rstrip("/") + f".avg_{os.path.basename(out_dir)}.ms"
    if os.path.exists(averaged):
        shutil.rmtree(averaged)

    kwargs = dict(
        vis=ms_path,
        outputvis=averaged,
        datacolumn="corrected",
        timeaverage=True,
        timebin=timebin,
        keepflags=False,
    )
    if chanbin:
        kwargs.update(chanaverage=True, chanbin=chanbin)
    mstransform(**kwargs)

    ms = ms_tool()
    ms.open(averaged)
    data = ms.getdata(["data", "u", "v", "weight", "axis_info"])
    ms.close()

    vis = data["data"].mean(axis=0)  # average polarizations -> (nchan, nrow)
    weight = data["weight"].mean(axis=0)  # (nrow,)
    freqs = data["axis_info"]["freq_axis"]["chan_freq"].ravel()

    c = 299792458.0
    u = np.concatenate([data["u"] * f / c for f in freqs])
    v = np.concatenate([data["v"] * f / c for f in freqs])
    vis_flat = vis.ravel()
    sigma = np.concatenate([1.0 / np.sqrt(weight) for _ in freqs])

    os.makedirs(out_dir, exist_ok=True)

    from astropy.io import fits

    fits.writeto(
        os.path.join(out_dir, "data.fits"),
        np.stack([vis_flat.real, vis_flat.imag], axis=-1).astype("float64"),
        overwrite=True,
    )
    fits.writeto(
        os.path.join(out_dir, "noise_map.fits"),
        np.stack([sigma, sigma], axis=-1).astype("float64"),
        overwrite=True,
    )
    fits.writeto(
        os.path.join(out_dir, "uv_wavelengths.fits"),
        np.stack([u, v], axis=-1).astype("float64"),
        overwrite=True,
    )

    shutil.rmtree(averaged)
    print(f"{out_dir}: {vis_flat.size} visibilities")
    return vis_flat.size


if __name__ == "__main__":
    ms_path = sys.argv[-1]
    base = os.path.dirname(os.path.abspath(__file__))
    for folder, timebin, chanbin in LEVELS:
        export_level(ms_path, os.path.join(base, folder), timebin, chanbin)
