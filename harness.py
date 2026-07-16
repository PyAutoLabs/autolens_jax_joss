"""
Timing and report harness for the PyAutoLens-JAX JOSS benchmarks.

Every benchmark in `benchmarks/` reports the four numbers required by the
PyAutoLens-JAX paper:

- **Total wall-clock time** — everything from benchmark start to finish,
  including dataset loading, model composition, JIT compilation and the fit.
- **JAX compilation time** — measured directly by compiling the same
  jit(vmap(value_and_grad(likelihood))) program the search driver builds and
  timing its first call against warm repeat calls.
- **Number of likelihood evaluations** — from the search's own accounting
  (e.g. ``n_starts * n_steps`` for a multi-start gradient search).
- **Post-compilation runtime** — the search wall-clock minus the measured
  compilation time.

Results are written to ``results/<benchmark>.json`` and the summary table
``results/RESULTS.md`` is regenerated from all committed JSONs. Quick-mode
runs (``--quick``) go to ``results/quick/`` (gitignored) so smoke tests never
pollute the official table.

Datasets are never committed to this repository. They are fetched on first
run from the ``autolens_workspace`` repository (or another public URL) and
cached under ``dataset/`` (gitignored) — this guarantees each benchmark runs
on byte-identical data to its paired workspace example.
"""

import json
import platform
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import os

WORKSPACE_BRANCH = os.environ.get("AUTOLENS_WORKSPACE_BRANCH", "main")
WORKSPACE_RAW_URL = (
    "https://raw.githubusercontent.com/PyAutoLabs/autolens_workspace/"
    f"{WORKSPACE_BRANCH}/dataset"
)

REPO_ROOT = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "results"
QUICK_RESULTS_DIR = RESULTS_DIR / "quick"


def fetch_workspace_dataset(relative_path: str, files: list) -> Path:
    """
    Fetch dataset files from the public autolens_workspace repository on first
    run, caching them under ``dataset/<relative_path>/``.

    Using the workspace's own dataset files guarantees the benchmark runs on
    byte-identical data to its paired ``start_here.py`` example.
    """
    target_dir = DATASET_DIR / relative_path
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_name in files:
        target = target_dir / file_name
        if target.exists():
            continue
        url = f"{WORKSPACE_RAW_URL}/{relative_path}/{file_name}"
        print(f"[harness] downloading {url}")
        urllib.request.urlretrieve(url, target)

    return target_dir


def fetch_url(url: str, relative_path: str, file_name: str) -> Path:
    """
    Fetch a single file from an arbitrary public URL (e.g. a Zenodo deposit),
    cached under ``dataset/<relative_path>/<file_name>``.
    """
    target_dir = DATASET_DIR / relative_path
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / file_name
    if not target.exists():
        print(f"[harness] downloading {url}")
        urllib.request.urlretrieve(url, target)
    return target


def _device_info() -> dict:
    import jax

    devices = jax.devices()
    return {
        "backend": devices[0].platform,
        "device": devices[0].device_kind,
        "device_count": len(devices),
        "hostname": platform.node(),
        "python": platform.python_version(),
    }


def _versions() -> dict:
    versions = {}
    for package in ("jax", "autoconf", "autofit", "autoarray", "autogalaxy", "autolens"):
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[package] = "not installed"
    return versions


class Benchmark:
    """
    Times one benchmark end-to-end and writes its JSON record.

    Usage::

        bench = Benchmark(name="imaging", paired_example="scripts/imaging/start_here.py",
                          description="...", quick=args.quick)
        with bench.phase("dataset"):
            ...
        bench.measure_compile(model=model, analysis=analysis, batch_size=16)
        result = bench.run_search(search=search, model=model, analysis=analysis,
                                  n_likelihood_evals=16 * 300)
        bench.finish(search_info={...})
    """

    def __init__(self, name: str, paired_example: str, description: str, quick: bool = False):
        self.name = name
        self.paired_example = paired_example
        self.description = description
        self.quick = quick
        self.phases = {}
        self.record = {}
        self._t_start = time.perf_counter()

    @contextmanager
    def phase(self, name: str):
        t0 = time.perf_counter()
        yield
        self.phases[name] = round(time.perf_counter() - t0, 3)

    def measure_compile(
        self,
        model,
        analysis,
        n_parallel: int = 16,
        map_batch_size=None,
        n_repeats: int = 3,
        use_grad: bool = True,
    ):
        """
        Measure JAX compilation time by building the same batched
        ``value_and_grad(likelihood)`` program the gradient search driver
        builds — ``jit(vmap(...))``, or ``jit(lax.map(..., batch_size))`` when
        memory-bounded batching is requested — and timing its first
        (compile + eval) call against the mean of ``n_repeats`` warm calls.

        Pass the search's own ``n_starts`` / ``batch_size`` so the compiled
        program (and its memory footprint) matches the fit exactly. For a
        sampling search (no gradients) pass ``use_grad=False`` to compile the
        plain batched likelihood instead, with ``n_parallel`` set to the
        sampler's batch size. The search's fit recompiles its private closure,
        so total wall-clock includes an equivalent compile; this measurement
        makes that component explicit and reproducible.
        """
        import jax
        import jax.numpy as jnp
        from autofit.non_linear.fitness import Fitness

        fitness = Fitness(
            model=model,
            analysis=analysis,
            fom_is_log_likelihood=False,
            resample_figure_of_merit=-np.inf,
            convert_to_chi_squared=True,
        )

        rng = np.random.default_rng(1)
        unit_vectors = rng.uniform(0.3, 0.7, size=(n_parallel, model.prior_count))
        params = jnp.asarray(
            [model.vector_from_unit_vector(unit_vector=u) for u in unit_vectors]
        )

        _inner = jax.value_and_grad(fitness.call) if use_grad else fitness.call

        if map_batch_size is None:
            batched = jax.jit(jax.vmap(_inner))
        else:

            @jax.jit
            def batched(p):
                return jax.lax.map(_inner, p, batch_size=map_batch_size)

        t0 = time.perf_counter()
        jax.block_until_ready(batched(params))
        t_first = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            jax.block_until_ready(batched(params))
        t_warm = (time.perf_counter() - t0) / n_repeats

        self.record["compile_s"] = round(t_first - t_warm, 3)
        self.record["warm_batch_eval_s"] = round(t_warm, 4)
        self.record["compile_measure_n_parallel"] = n_parallel
        self.record["compile_measure_map_batch_size"] = map_batch_size
        print(
            f"[harness] compile: {self.record['compile_s']}s "
            f"(warm batch of {n_parallel}: {t_warm:.3f}s)"
        )

    def run_search(self, search, model, analysis, n_likelihood_evals=None):
        """
        Run ``search.fit`` under a wall-clock timer. Pass
        ``n_likelihood_evals`` where the search's accounting is known (e.g.
        ``n_starts * n_steps``); otherwise it is taken from the result's
        samples where possible.
        """
        t0 = time.perf_counter()
        result = search.fit(model=model, analysis=analysis)
        self.record["search_wall_s"] = round(time.perf_counter() - t0, 3)

        if n_likelihood_evals is None:
            try:
                samples = result[0].samples if isinstance(result, list) else result.samples
                n_likelihood_evals = len(samples.log_likelihood_list)
            except (AttributeError, TypeError):
                n_likelihood_evals = None
        self.record["n_likelihood_evals"] = n_likelihood_evals

        try:
            samples = result[0].samples if isinstance(result, list) else result.samples
            self.record["max_log_likelihood"] = round(float(max(samples.log_likelihood_list)), 2)
        except (AttributeError, TypeError, ValueError):
            pass

        return result

    def finish(self, **extra):
        """Write ``results/<name>.json`` and regenerate ``results/RESULTS.md``."""
        total_s = time.perf_counter() - self._t_start
        compile_s = self.record.get("compile_s")
        search_s = self.record.get("search_wall_s")

        self.record.update(
            {
                "benchmark": self.name,
                "paired_example": self.paired_example,
                "description": self.description,
                "quick": self.quick,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "total_wall_s": round(total_s, 3),
                "phases": self.phases,
                "hardware": _device_info(),
                "versions": _versions(),
            }
        )
        if compile_s is not None and search_s is not None:
            self.record["post_compile_s"] = round(search_s - compile_s, 3)
        self.record.update(extra)

        out_dir = QUICK_RESULTS_DIR if self.quick else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self.name}.json"
        with open(out_path, "w") as f:
            json.dump(self.record, f, indent=2)
        print(f"[harness] wrote {out_path}")

        if not self.quick:
            regenerate_results_md()

        minutes = total_s / 60.0
        print(
            f"[harness] {self.name}: total {minutes:.2f} min | "
            f"compile {compile_s}s | search {search_s}s | "
            f"evals {self.record.get('n_likelihood_evals')} | "
            f"max logL {self.record.get('max_log_likelihood')}"
        )


def regenerate_results_md():
    """Rebuild ``results/RESULTS.md`` from every official ``results/*.json``."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path) as f:
            rows.append(json.load(f))

    lines = [
        "# PyAutoLens-JAX benchmark results",
        "",
        "Generated by `harness.py` from the JSON records in this directory —",
        "do not edit by hand. Quick-mode runs are excluded (they live in",
        "`results/quick/`, gitignored).",
        "",
        "| Benchmark | Paired example | Device | Search | Likelihood evals |"
        " Compile (s) | Post-compile (s) | Total (min) | Max logL | Date |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        hw = r.get("hardware", {})
        search_info = r.get("search_info", {})
        lines.append(
            "| {benchmark} | `{paired}` | {device} | {search} | {evals} |"
            " {compile} | {post} | {total:.2f} | {logl} | {date} |".format(
                benchmark=r.get("benchmark"),
                paired=r.get("paired_example"),
                device=hw.get("device", "?"),
                search=search_info.get("type", "?"),
                evals=r.get("n_likelihood_evals", "?"),
                compile=r.get("compile_s", "?"),
                post=r.get("post_compile_s", "?"),
                total=r.get("total_wall_s", 0) / 60.0,
                logl=r.get("max_log_likelihood", "?"),
                date=r.get("date", "?"),
            )
        )
    lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))
    print(f"[harness] regenerated {RESULTS_DIR / 'RESULTS.md'}")
