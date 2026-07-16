# SDP.81 ALMA visibility preparation

The interferometry benchmarks fit the ALMA Science Verification observations
of **SDP.81** (H-ATLAS J090311.6+003906), the z = 3.042 Einstein ring lensed
by a z = 0.299 elliptical, from the 2014 ALMA Long Baseline Campaign
(ALMA Partnership et al. 2015, ApJ 808, L4). The calibrated measurement sets
are public on the ALMA Science Portal:

    https://almascience.org/alma-data/science-verification
    (project 2011.0.00016.SV, "SDP.81 Band 6/7 calibrated data")

They are tens of GB and require CASA, so this repository does not download
them automatically. Instead:

1. Download the Band 6 continuum calibrated measurement set from the portal.
2. Run `export_uv.py` inside CASA (`casa -c export_uv.py <ms_path>`): it
   time- and frequency-averages conservatively (keeping > 1 million
   visibilities), extracts the calibrated continuum visibilities, and writes
   the three PyAutoLens input FITS files:
   - `data.fits` — real/imaginary visibilities,
   - `noise_map.fits` — per-visibility RMS from the measurement-set weights,
   - `uv_wavelengths.fits` — u,v coordinates in wavelengths.
3. Place them in `dataset/interferometer/sdp81/` (or upload them to a Zenodo
   deposit and set `SDP81_URL` in `benchmarks/interferometer.py`, which then
   fetches them automatically like every other benchmark).

Until the products exist, `benchmarks/interferometer.py` and
`benchmarks/imaging_and_interferometer.py` exit with a pointer to this file.
The paired workspace example (`scripts/interferometer/start_here.py`) is
upgraded to SDP.81 in the same step — tracked in
https://github.com/PyAutoLabs/autolens_workspace/issues/281.
