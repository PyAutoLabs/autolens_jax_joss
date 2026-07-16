# autolens_jax_joss

Runnable benchmarks for the **PyAutoLens-JAX** paper: every performance claim
in the paper's *End-to-end modelling benchmarks* section corresponds to one
script in [`benchmarks/`](benchmarks), which fits **real data** with a
differentiable, GPU-accelerated JAX likelihood and reports its runtime.

Every benchmark records the four numbers the paper requires — **total
wall-clock time**, **JAX compilation time**, **number of likelihood
evaluations** and **post-compilation runtime** — plus hardware and package
versions, into `results/<benchmark>.json`. The official summary table is
[`results/RESULTS.md`](results/RESULTS.md), regenerated from those JSONs.

## Pairing with autolens_workspace

Each benchmark is paired to a user-facing example in
[`autolens_workspace`](https://github.com/PyAutoLabs/autolens_workspace) and
runs on the **same real dataset** (fetched from the workspace repository on
first run, so the data is byte-identical). The workspace example teaches the
workflow; the benchmark times it.

| Benchmark | Script | Paired workspace example | Real dataset |
|---|---|---|---|
| Galaxy-scale CCD imaging | `benchmarks/imaging.py` | `scripts/imaging/start_here.py` | JWST COSMOS-Web Ring F150W |
| Interferometry | `benchmarks/interferometer.py` | `scripts/interferometer/start_here.py` | ALMA SDP.81 (>1M visibilities) |
| Point-source lensing | `benchmarks/point_source.py` | `scripts/point_source/start_here.py` | RXJ1131-1231 (positions + time delays) |
| Group-scale lensing | `benchmarks/group.py` | `scripts/group/start_here.py` | real group-scale lens |
| Cluster-scale lensing | `benchmarks/cluster.py` | `scripts/cluster/start_here.py` | Abell 2744 (multiple images) |
| Weak lensing | `benchmarks/weak.py` | `scripts/weak/start_here.py` | Abell 2744 JWST shape catalogue |
| Multi-band imaging | `benchmarks/multi_band.py` | `scripts/multi/start_here.py` | JWST COSMOS-Web Ring (4 bands) |
| Strong + weak lensing | `benchmarks/strong_and_weak.py` | `scripts/weak/features/strong_lensing/` | Abell 2744 |
| Imaging + point source | `benchmarks/imaging_and_point_source.py` | `scripts/multi/features/imaging_and_point_source/` | RXJ1131-1231 |
| Imaging + interferometry | `benchmarks/imaging_and_interferometer.py` | `scripts/multi/features/imaging_and_interferometer/` | SDP.81 |

Benchmarks not yet listed in `benchmarks/` are being added phase by phase —
see the tracking issue
([autolens_workspace#281](https://github.com/PyAutoLabs/autolens_workspace/issues/281)).

## Installation

```bash
pip install "autolens[jax]"
git clone https://github.com/PyAutoLabs/autolens_jax_joss
cd autolens_jax_joss
```

A GPU is strongly recommended for the official timings (the paper's numbers
are from an NVIDIA A100), but every script also runs on CPU or a small GPU.

## Running a benchmark

```bash
python benchmarks/imaging.py             # full benchmark
python benchmarks/imaging.py --quick     # fast smoke run (any machine; results go to results/quick/)
python benchmarks/imaging.py --search nautilus   # gradient-free nested-sampling baseline
```

Datasets are **never stored in this repository**: each script downloads its
data on first run (from the `autolens_workspace` repository or another public
archive URL) and caches it under `dataset/` (gitignored).

The default search is **multi-start Adam** — many broad gradient-descent
starts run in parallel on the GPU with the best kept, the robust-and-fast
recipe for differentiable lens likelihoods (a single cold start reliably
lands in the wrong basin; see the workspace's
`guides/modeling/searches.py`).

## Reading the results

`results/RESULTS.md` has one row per benchmark. Timing semantics:

- **Compile (s)** — measured by compiling the same
  `jit(vmap(value_and_grad(likelihood)))` program the search driver builds
  and subtracting warm-call time from the first call.
- **Post-compile (s)** — search wall-clock minus the measured compile time:
  the steady-state inference cost.
- **Total (min)** — everything, including dataset download-cache hits,
  model composition, compilation and the fit.

## Relation to the paper

This repository is the reproducibility companion to *PyAutoLens-JAX:
Differentiable GPU-accelerated strong and weak lensing from galaxies to
clusters*. The paper's benchmark section cites the numbers in
`results/RESULTS.md` directly.
