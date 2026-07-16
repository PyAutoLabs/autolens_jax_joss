"""
CASA export script: SDP.81 calibrated measurement set -> PyAutoLens FITS.

Run inside CASA (versions 6.x):

    casa -c export_uv.py <path_to_calibrated_ms>

Writes data.fits (real/imag visibilities), noise_map.fits (per-visibility RMS
from the MS weights) and uv_wavelengths.fits (u,v in wavelengths) alongside
this script. Averaging keeps well over one million visibilities; adjust
`timebin` / `width` below only if you know you want a different size.
"""

import sys

import numpy as np


def export(ms_path):
    from casatools import ms as ms_tool
    from casatasks import mstransform

    averaged = ms_path.rstrip("/") + ".averaged.ms"
    mstransform(
        vis=ms_path,
        outputvis=averaged,
        datacolumn="corrected",
        timeaverage=True,
        timebin="30s",
        chanaverage=True,
        chanbin=4,
        keepflags=False,
    )

    ms = ms_tool()
    ms.open(averaged)
    data = ms.getdata(["data", "u", "v", "weight", "axis_info"])
    ms.close()

    # Average polarizations; flatten channels x rows into one visibility list.
    vis = data["data"].mean(axis=0)  # (nchan, nrow)
    weight = data["weight"].mean(axis=0)  # (nrow,)
    freqs = data["axis_info"]["freq_axis"]["chan_freq"].ravel()  # (nchan,)

    c = 299792458.0
    u = np.concatenate([data["u"] * f / c for f in freqs])
    v = np.concatenate([data["v"] * f / c for f in freqs])
    vis_flat = vis.ravel()
    sigma = np.concatenate([1.0 / np.sqrt(weight) for _ in freqs])

    print(f"total visibilities: {vis_flat.size}")

    from astropy.io import fits

    fits.writeto(
        "data.fits",
        np.stack([vis_flat.real, vis_flat.imag], axis=-1).astype("float64"),
        overwrite=True,
    )
    fits.writeto(
        "noise_map.fits",
        np.stack([sigma, sigma], axis=-1).astype("float64"),
        overwrite=True,
    )
    fits.writeto(
        "uv_wavelengths.fits",
        np.stack([u, v], axis=-1).astype("float64"),
        overwrite=True,
    )


if __name__ == "__main__":
    export(sys.argv[-1])
