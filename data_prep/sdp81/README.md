# SDP.81 ALMA visibility preparation

The interferometry benchmarks fit the ALMA Science Verification observations
of **SDP.81** (H-ATLAS J090311.6+003906), the z = 3.042 Einstein ring lensed
by a z = 0.299 elliptical, from the 2014 ALMA Long Baseline Campaign
(ALMA Partnership et al. 2015, ApJ 808, L4). The calibrated measurement sets
are public on the ALMA Science Portal:

    https://almascience.org/alma-data/science-verification
    (project 2011.0.00016.SV, "SDP.81 Band 6/7 calibrated data")

## Design: one export, three averaging levels

Time/channel-averaging reduces visibility count while **keeping the long
baselines** — an averaged SDP.81 continuum uv-table is still genuinely
~25-30 mas resolution. The export therefore emits the same dataset at three
sizes into sibling folders:

| Folder | Averaging | ~Visibilities | Role |
|---|---|---|---|
| `sdp81/` | heavy (60 s, all channels) | ~50k | **default** — the user-example dataset paired with `interferometer/start_here.py` (a few MB; auto-downloadable) |
| `sdp81_mid/` | moderate (10 s, chanbin 4) | ~500k | scaling point |
| `sdp81_full/` | light (2 s, no chanbin) | >1M | scaling point — the paper's ">1 million visibilities" row |

The real-space mask is 0.025"/pixel in every case — set by the array's actual
resolution, not the visibility count. `benchmarks/interferometer.py --nvis
{default,mid,full}` runs each level; the resulting table documents that
per-evaluation runtime is ~flat in N_vis (the likelihood is dominated by the
real-space grid and transform setup), which is the honest form of the paper's
visibility-scaling claim.

## Producing the products

1. Download the Band 6 continuum calibrated measurement set from the portal
   (tens of GB — this is the one heavyweight, one-off step).
2. Run `export_uv.py` inside CASA (`casa -c export_uv.py <ms_path>`), or with
   pip-installed `casatools`/`casatasks` (`python export_uv.py <ms_path>`).
   It writes all three levels' `data.fits` / `noise_map.fits` /
   `uv_wavelengths.fits`.
3. Place the folders under `dataset/interferometer/`, or host the (small)
   default level at a public URL and set `SDP81_URL` in
   `benchmarks/interferometer.py` — the workspace example uses the same
   auto-download.

Until the products exist, the two interferometer benchmarks exit with a
pointer to this file; the paired `scripts/interferometer/start_here.py`
upgrade tracks in https://github.com/PyAutoLabs/autolens_workspace/issues/281.
