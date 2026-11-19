"""Campaign driver: sweep resolutions, reduce to statistics, archive."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy
from numpy.typing import NDArray
from scipy.stats import kurtosis, skew

from coastline_config import RunConfig
from coastline_estimators import (
    NESTED_BLOCK, closed_ruler_limit, nested_block_seed, polyline_ruler_limit, random_points,
    random_points_nested, yardstick,
)
from coastline_geometry import CoastlineModel

__all__ = ["ResolutionResult", "summarise", "run_campaign", "richardson_dimension"]

_ARCHIVE_VERSION = 9  # 9: explicit open/closed topology and periodic RP/YS

_STAT_NAMES = (
    "mean", "std", "median", "skewness", "kurtosis",
    "q1", "q3", "p2_5", "p97_5", "minimum", "maximum",
)


@dataclass
class ResolutionResult:
    resolution: float
    stats: Dict[str, float]
    hist_counts: NDArray[np.int64]
    hist_density: NDArray[np.float64]
    bin_edges: NDArray[np.float64]
    hist_is_pointmass: bool
    hist_pointmass: float
    n_points_stats: Dict[str, float]
    length_samples: NDArray[np.float64] | None = None
    n_points_samples: NDArray[np.int64] | None = None


def summarise(lengths: NDArray[np.float64]) -> Dict[str, float]:
    p2_5, q1, q3, p97_5 = np.percentile(lengths, [2.5, 25.0, 75.0, 97.5])
    return {
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths, ddof=1)),
        "median": float(np.median(lengths)),
        "skewness": float(skew(lengths)),
        "kurtosis": float(kurtosis(lengths)),
        "q1": float(q1),
        "q3": float(q3),
        "p2_5": float(p2_5),
        "p97_5": float(p97_5),
        "minimum": float(np.min(lengths)),
        "maximum": float(np.max(lengths)),
    }


def _summarise_n_points(n_points: NDArray[np.int64]) -> Dict[str, float]:
    return {
        "mean": float(np.mean(n_points)),
        "std": float(np.std(n_points, ddof=1)) if n_points.size > 1 else 0.0,
        "minimum": float(np.min(n_points)),
        "maximum": float(np.max(n_points)),
    }


def _histogram_pdf(
    lengths: NDArray[np.float64], n_bins: int
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], bool, float]:
    """Histograms for regular distributions; explicit delta masses when unresolved.

    NumPy's default fixed-bin histogram cannot represent ranges of just a few
    ulps. In such a case the distribution is numerically indistinguishable from
    a point mass. The archive records this *as a delta* so MC-fit never turns
    a deterministic length into artificial uniform-in-bin sampling noise.
    """
    point = float(np.mean(lengths))
    tolerance = 16.0 * n_bins * np.finfo(float).eps * max(1.0, abs(point))
    is_pointmass = bool(np.ptp(lengths) <= tolerance)
    if is_pointmass:
        # A visualisation-only bin around the point. It is *not* used for
        # sampling; see hist_is_pointmass and hist_pointmass in the archive.
        half_width = max(1e-8 * max(abs(point), 1.0), 16.0 * tolerance)
        edges = np.linspace(point - half_width, point + half_width, n_bins + 1)
        counts = np.zeros(n_bins, dtype=np.int64)
        counts[n_bins // 2] = lengths.size
    else:
        counts, edges = np.histogram(lengths, bins=n_bins)
    widths = np.diff(edges)
    density = counts.astype(float) / (lengths.size * widths)
    return counts.astype(np.int64), density, edges.astype(float), is_pointmass, point


def _seed_for_resolution(seed: int, method: str, resolution: float) -> np.random.SeedSequence:
    method_code = 0x5250 if method == "rp" else 0x5953
    if method == "rp":
        key = int(round(resolution)) & 0xFFFFFFFFFFFFFFFF
    else:
        key = int(np.float64(resolution).view(np.uint64))
    lo = key & 0xFFFFFFFF
    hi = (key >> 32) & 0xFFFFFFFF
    return np.random.SeedSequence([int(seed), method_code, lo, hi])


def _reduce(
    cfg: RunConfig,
    resolution: float,
    lengths: NDArray[np.float64],
    n_points: NDArray[np.int64],
    keep_lengths: bool,
    keep_n_points: bool,
) -> ResolutionResult:
    hist_counts, hist_density, edges, is_pointmass, point = _histogram_pdf(lengths, cfg.n_bins)
    return ResolutionResult(
        resolution=float(resolution),
        stats=summarise(lengths),
        hist_counts=hist_counts,
        hist_density=hist_density,
        bin_edges=edges,
        hist_is_pointmass=is_pointmass,
        hist_pointmass=point,
        n_points_stats=_summarise_n_points(n_points),
        length_samples=lengths if keep_lengths else None,
        n_points_samples=n_points if keep_n_points else None,
    )


def _rp_independent_one(cfg: RunConfig, model: CoastlineModel, resolution: float) -> ResolutionResult:
    rng = np.random.default_rng(_seed_for_resolution(cfg.seed, "rp", resolution))
    n_p = int(resolution)
    lengths = random_points(
        model, n_p, cfg.n_realisations, rng,
        uniform_in=cfg.uniform_in, max_elements=cfg.chunk_elements, engine=cfg.engine,
    )
    n_points = np.full(cfg.n_realisations, n_p, dtype=np.int64)
    return _reduce(cfg, resolution, lengths, n_points, cfg.save_samples, False)


def _rp_nested_block(
    cfg: RunConfig, model: CoastlineModel, block: int, resolutions: NDArray[np.int64]
) -> NDArray[np.float64]:
    """One independently seeded block: results do not depend on n_jobs or chunk size."""
    lo = block * NESTED_BLOCK
    rows = min(NESTED_BLOCK, cfg.n_realisations - lo)
    rng = np.random.default_rng(nested_block_seed(cfg.seed, block))
    return random_points_nested(
        model, resolutions, rows, rng,
        uniform_in=cfg.uniform_in, max_elements=cfg.chunk_elements, engine=cfg.engine,
    )


_YS_COMMON_CODE = 0x434F4D  # "COM"


def _ys_seeds(cfg: RunConfig, model: CoastlineModel, ruler: float) -> NDArray[np.float64]:
    """Seed stations of one ruler, drawn up front so the realisations can be split
    across processes without changing them.

    ``independent``: each ruler has its own stream (seeded by the ruler).
    ``common``: one stream for the whole campaign, so realisation k starts every
    ruler from the same station; each ruler still sees uniform seed stations, i.e.
    exactly the same P(L | d), but column k of the samples is one whole survey.
    """
    if cfg.ys_seeds == "common":
        ss = np.random.SeedSequence([int(cfg.seed), 0x5953, _YS_COMMON_CODE])
    else:
        ss = _seed_for_resolution(cfg.seed, "ys", ruler)
    return np.random.default_rng(ss).random(cfg.n_realisations) * model.length


def _ys_directions(cfg: RunConfig, ruler: float) -> NDArray[np.int64]:
    """Independent fair orientations, shared across rulers with common seed stations."""
    if cfg.ys_seeds == "common":
        ss = np.random.SeedSequence([int(cfg.seed), 0x5953, _YS_COMMON_CODE, 0x444952])
    else:
        # A distinct substream from the station seed sequence at this ruler.
        key = int(np.float64(ruler).view(np.uint64))
        ss = np.random.SeedSequence([int(cfg.seed), 0x5953, 0x444952,
                                     key & 0xFFFFFFFF, key >> 32])
    return (2 * np.random.default_rng(ss).integers(0, 2, cfg.n_realisations) - 1).astype(np.int64)


def _ys_piece(
    cfg: RunConfig, model: CoastlineModel, ruler: float, s0: NDArray[np.float64],
    directions: NDArray[np.int64] | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    return yardstick(
        model, ruler, s0=s0, end_policy=cfg.end_policy,
        max_elements=cfg.chunk_elements, engine=cfg.engine, walker=cfg.ys_walker,
        directions=directions,
    )


def _warm_up(cfg: RunConfig, model: CoastlineModel) -> None:
    """Compile (or load from cache) the numba kernels once, before any worker starts."""
    if cfg.engine != "numba":
        return
    rng = np.random.default_rng(0)
    if cfg.method == "rp":
        random_points(model, 4, 2, rng, uniform_in=cfg.uniform_in, engine="numba")
        random_points_nested(model, [3, 4], 2, rng, uniform_in=cfg.uniform_in, engine="numba")
    else:
        # A ruler that both walkers accept on any curve (the polyline walker has an
        # upper limit); its value does not matter for compilation.
        ruler = (min(float(cfg.ys_start), 0.5 * polyline_ruler_limit(model))
                 if cfg.topology == "closed"
                 else min(0.5 * model.length, 0.5 * polyline_ruler_limit(model)))
        yardstick(
            model, ruler, s0=np.array([0.5 * model.length]), engine="numba",
            walker=cfg.ys_walker,
            end_policy=cfg.end_policy,
            directions=np.array([1], dtype=np.int64) if cfg.topology == "closed" else None,
        )


def _parallel(cfg: RunConfig, verbose: bool):
    from joblib import Parallel

    return Parallel(n_jobs=cfg.n_jobs, backend="loky", verbose=5 if verbose else 0)


def _run_rp_independent(cfg, model, resolutions, verbose) -> List[ResolutionResult]:
    if cfg.n_jobs == 1:
        results: List[ResolutionResult] = []
        for i, res in enumerate(resolutions, 1):
            results.append(_rp_independent_one(cfg, model, float(res)))
            if verbose:
                print(
                    f"  [{i}/{resolutions.size}] {cfg.resolution_label}={res:g} "
                    f"<L>={results[-1].stats['mean']:.5f} km",
                    flush=True,
                )
        return results
    from joblib import delayed

    # Cost grows with n_points: hand out the largest first so that no long task is
    # left for the end (longest-processing-time-first scheduling).
    order = np.argsort(-resolutions, kind="stable")
    out = _parallel(cfg, verbose)(
        delayed(_rp_independent_one)(cfg, model, float(resolutions[k])) for k in order
    )
    results: List[ResolutionResult] = [None] * resolutions.size  # type: ignore[list-item]
    for k, r in zip(order, out):
        results[k] = r
    return results


def _run_rp_nested(cfg, model, resolutions, verbose) -> List[ResolutionResult]:
    n_blocks = -(-cfg.n_realisations // NESTED_BLOCK)
    if verbose:
        print(
            f"  nested sweep: {n_blocks} block(s) of up to {NESTED_BLOCK} realisations, "
            f"n_points up to {int(resolutions.max())}",
            flush=True,
        )
    if cfg.n_jobs == 1:
        blocks = [_rp_nested_block(cfg, model, b, resolutions) for b in range(n_blocks)]
    else:
        from joblib import delayed

        blocks = _parallel(cfg, verbose)(
            delayed(_rp_nested_block)(cfg, model, b, resolutions) for b in range(n_blocks)
        )
    lengths = np.concatenate(blocks, axis=0)          # (n_realisations, n_res)
    results = []
    for k, res in enumerate(resolutions):
        col = np.ascontiguousarray(lengths[:, k])
        n_points = np.full(cfg.n_realisations, int(res), dtype=np.int64)
        results.append(_reduce(cfg, float(res), col, n_points, cfg.save_samples, False))
    return results


def _run_ys(cfg, model, resolutions, verbose) -> List[ResolutionResult]:
    if cfg.n_jobs == 1:
        pieces = 1
    else:
        from joblib import effective_n_jobs

        pieces = max(1, min(effective_n_jobs(cfg.n_jobs), cfg.n_realisations // 64))

    if cfg.ys_seeds == "common":
        common = _ys_seeds(cfg, model, 0.0)
        seeds = [common] * resolutions.size
    else:
        seeds = [_ys_seeds(cfg, model, float(r)) for r in resolutions]
    directions = None
    if cfg.topology == "closed":
        if cfg.ys_seeds == "common":
            common_dirs = _ys_directions(cfg, 0.0)
            directions = [common_dirs] * resolutions.size
        else:
            directions = [_ys_directions(cfg, float(r)) for r in resolutions]
    # Smallest ruler = most steps: schedule it first, and split every ruler into
    # `pieces` so that a single ruler cannot hold the whole campaign back.
    order = np.argsort(resolutions, kind="stable")
    tasks = [
        (int(k), part)
        for k in order
        for part in np.array_split(np.arange(cfg.n_realisations), pieces)
        if part.size
    ]
    if cfg.n_jobs == 1:
        outs = []
        for i, (k, part) in enumerate(tasks, 1):
            outs.append(_ys_piece(cfg, model, float(resolutions[k]), seeds[k][part],
                                  None if directions is None else directions[k][part]))
            if verbose:
                print(
                    f"  [{i}/{len(tasks)}] {cfg.resolution_label}={resolutions[k]:g} "
                    f"<L>={outs[-1][0].mean():.5f} km",
                    flush=True,
                )
    else:
        from joblib import delayed

        outs = _parallel(cfg, verbose)(
            delayed(_ys_piece)(cfg, model, float(resolutions[k]), seeds[k][part],
                               None if directions is None else directions[k][part])
            for k, part in tasks
        )

    gathered: Dict[int, list] = {}
    for (k, part), o in zip(tasks, outs):
        gathered.setdefault(k, []).append((part, o))
    results = []
    for k, res in enumerate(resolutions):
        lengths = np.empty(cfg.n_realisations)
        n_points = np.empty(cfg.n_realisations, dtype=np.int64)
        for part, (l_, n_) in gathered[k]:
            lengths[part] = l_
            n_points[part] = n_
        results.append(_reduce(cfg, float(res), lengths, n_points, True, True))
    return results


def richardson_dimension(
    resolutions: NDArray[np.float64],
    mean_lengths: NDArray[np.float64],
    method: str,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    ell = (
        mean_lengths / np.maximum(resolutions - 1.0, 1.0)
        if method == "rp"
        else np.asarray(resolutions, dtype=float)
    )
    if ell.size < 2:
        # A slope needs two resolutions; np.gradient raises on a single one.
        return ell, np.full(ell.size, np.nan)
    order = np.argsort(-ell)
    e, m = ell[order], mean_lengths[order]
    edge_order = 2 if ell.size >= 3 else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.gradient(np.log(m), np.log(e), edge_order=edge_order)
    d_local = np.empty(ell.size, dtype=float)
    d_local[order] = 1.0 - slope
    return ell, d_local


def _metadata(cfg: RunConfig, model: CoastlineModel, wall_seconds: float) -> Dict[str, Any]:
    return {
        "archive_version": _ARCHIVE_VERSION,
        "purpose": "empirical sampling distributions for downstream Monte-Carlo fitting",
        "config": cfg.to_dict(),
        "reference_length_km": model.length,
        "polyline_length_km": model.polyline_length,
        "n_input_samples": int(model.lat_rad.size),
        "topology": cfg.topology,
        "closure_added": model.closure_added,
        "closure_gap_km": model.closure_gap_km,
        "n_points_definition": (
            "closed polygon vertices including repeated starting station; "
            "n_points-1 measured chords" if cfg.topology == "closed"
            else "open polyline vertices; n_points-1 measured chords"
        ),
        "ys_closed_rule": (
            "one full lap, randomized orientation, one closing chord, no endpoint fragments"
            if cfg.topology == "closed" and cfg.method == "ys" else None
        ),
        "map": {
            "nodes": model.map_nodes,
            "passes": model.map_passes,
            "max_density_distortion": model.map_distortion,
            "max_density_distortion_resolvable": model.map_distortion_resolvable,
            "unresolved_arc_km": model.map_unresolved_km,
            "converged": model.map_converged,
        },
        "samples_dependent_across_resolutions": bool(
            (cfg.method == "rp" and cfg.rp_sweep == "nested")
            or (cfg.method == "ys" and cfg.ys_seeds == "common")
        ),
        "arc_lipschitz": (
            model.arc_lipschitz if cfg.method == "ys" and cfg.ys_walker == "march" else None
        ),
        "polyline_ruler_limit_km": (
            polyline_ruler_limit(model)
            if cfg.method == "ys" and cfg.ys_walker == "polyline" else None
        ),
        "closed_ruler_limit_km": (
            closed_ruler_limit(model)
            if cfg.method == "ys" and cfg.topology == "closed" else None
        ),
        "closed_ruler_limit_kind": (
            "certified_sufficient_lower_bound_not_exact_threshold"
            if cfg.method == "ys" and cfg.topology == "closed" else None
        ),
        "wall_seconds": wall_seconds,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }


def run_campaign(cfg: RunConfig, verbose: bool = True) -> Dict[str, Any]:
    t_start = time.perf_counter()

    model = CoastlineModel.from_file(
        cfg.input_file,
        cfg.input_format,
        kind=cfg.interpolation,
        topology=cfg.topology,
        quad_order=cfg.quad_order,
        subdivisions=cfg.subdivisions,
        map_tol=cfg.map_tol,
        map_max_nodes=cfg.map_max_nodes,
    )
    resolutions = cfg.resolutions
    if verbose:
        print(model)
        if cfg.topology == "closed":
            print("closed contour: " + ("added closing edge" if model.closure_added else "closing vertex already present") +
                  f"; original endpoint gap = {model.closure_gap_km:.6g} km")
        lat_deg, lon_deg = np.degrees(model.lat_rad), np.degrees(model.lon_rad)
        # A quick visual check that the columns were read in the right order.
        print(
            f"input ({cfg.input_format}): lat [{lat_deg.min():.5f}, {lat_deg.max():.5f}], "
            f"lon [{lon_deg.min():.5f}, {lon_deg.max():.5f}] deg"
        )
        print(
            f"reference length L_ref = {model.length:.6f} km "
            f"({cfg.interpolation} interpolant, {model.lat_rad.size} input samples)"
        )
        if not model.map_converged or model.map_unresolved_km > 0.0:
            print(
                f"s<->t map: max |q-1| = {model.map_distortion_resolvable:.1e} where "
                f"map_tol is attainable; {model.map_unresolved_km:.3e} km of arc left above "
                "map_tol (round-off floor or node cap)"
            )
        if cfg.method == "rp":
            sweep = f", {cfg.rp_sweep} sweep, {cfg.rp_spacing} spacing"
            if cfg.rp_spacing == "log":
                sweep += f", {cfg.rp_per_decade}/decade"
        else:
            sweep = f", {cfg.ys_seeds} seed stations, {cfg.ys_walker} walker"
        print(
            f"sweeping {resolutions.size} values of {cfg.resolution_label} "
            f"x {cfg.n_realisations} realisations (engine={cfg.engine}{sweep})"
        )
        if cfg.method == "rp":
            per_real = resolutions.max() if cfg.rp_sweep == "nested" else resolutions.sum()
            print(f"workload: {float(per_real) * cfg.n_realisations:.2e} station evaluations")

    if cfg.method == "ys" and cfg.ys_walker == "polyline":
        # Checked here, before any work: rulers are scheduled smallest (slowest) first,
        # so the estimator itself would only refuse the largest one at the very end.
        limit = polyline_ruler_limit(model)
        if not float(resolutions.max()) < limit:
            raise ValueError(
                f"ys_walker='polyline' is exact only for rulers below {limit:.6g} km on "
                f"this curve (its distance to the nearest pole); ys_stop = "
                f"{float(resolutions.max()):.6g} km. Lower ys_stop or use ys_walker='march'"
            )
        if cfg.topology == "closed":
            # Same reason, and evaluated here so that worker processes receive the
            # cached value with the model.
            limit = closed_ruler_limit(model, required=float(resolutions.max()))
            if not float(resolutions.max()) < limit:
                raise ValueError(
                    f"closed YS: ys_stop={float(resolutions.max()):.6g} km is not "
                    f"certified below the safe lower bound {limit:.6g} km. "
                    "The true critical ruler may be larger; lower ys_stop or "
                    "increase the contour resolution."
                )
    elif cfg.method == "ys":
        # Lazy property: evaluate it here, once, so that worker processes receive it
        # with the model instead of each recomputing it.
        model.arc_lipschitz
    _warm_up(cfg, model)
    if cfg.method == "ys":
        results = _run_ys(cfg, model, resolutions, verbose)
    elif cfg.rp_sweep == "nested":
        results = _run_rp_nested(cfg, model, resolutions, verbose)
    else:
        results = _run_rp_independent(cfg, model, resolutions, verbose)

    archive = _assemble(cfg, model, results, time.perf_counter() - t_start)
    _write(cfg, archive, verbose=verbose)
    return archive


def _assemble(
    cfg: RunConfig,
    model: CoastlineModel,
    results: List[ResolutionResult],
    wall: float,
) -> Dict[str, Any]:
    resolutions = np.array([r.resolution for r in results], dtype=float)
    stats = {k: np.array([r.stats[k] for r in results]) for k in _STAT_NAMES}
    ell, d_local = richardson_dimension(resolutions, stats["mean"], cfg.method)

    hist_counts = np.stack([r.hist_counts for r in results])
    hist_density = np.stack([r.hist_density for r in results])
    bin_edges = np.stack([r.bin_edges for r in results])
    hist_is_pointmass = np.array([r.hist_is_pointmass for r in results], dtype=bool)
    hist_pointmass = np.array([r.hist_pointmass for r in results], dtype=float)

    archive: Dict[str, Any] = {
        "archive_version": np.int64(_ARCHIVE_VERSION),
        "method": np.array(cfg.method),
        "topology": np.array(cfg.topology),
        "resolution_label": np.array(cfg.resolution_label),
        "resolution": resolutions,
        "n_realisations": np.int64(cfg.n_realisations),
        "n_bins": np.int64(cfg.n_bins),
        "length_hist_counts": hist_counts,
        "length_hist_density": hist_density,
        "length_bin_edges": bin_edges,
        "hist_is_pointmass": hist_is_pointmass,
        "hist_pointmass": hist_pointmass,
        "effective_ruler_km": ell,
        "richardson_dimension": d_local,
        "reference_length_km": np.float64(model.length),
        "polyline_length_km": np.float64(model.polyline_length),
        "map_max_density_distortion": np.float64(model.map_distortion),
        "n_points_mean": np.array([r.n_points_stats["mean"] for r in results]),
        "n_points_std": np.array([r.n_points_stats["std"] for r in results]),
        "n_points_min": np.array([r.n_points_stats["minimum"] for r in results]),
        "n_points_max": np.array([r.n_points_stats["maximum"] for r in results]),
        "engine": np.array(cfg.engine),
        "rp_sweep": np.array(cfg.rp_sweep if cfg.method == "rp" else "n/a"),
        "ys_seeds": np.array(cfg.ys_seeds if cfg.method == "ys" else "n/a"),
        "ys_walker": np.array(cfg.ys_walker if cfg.method == "ys" else "n/a"),
        "metadata_json": np.array(json.dumps(_metadata(cfg, model, wall))),
    }
    archive.update(stats)

    if cfg.method == "ys":
        length_samples = np.stack([r.length_samples for r in results])
        n_points_samples = np.stack([r.n_points_samples for r in results])
        archive["length_samples"] = length_samples
        archive["n_points_samples"] = n_points_samples

        n_min = int(n_points_samples.min())
        n_max = int(n_points_samples.max())
        support = np.arange(n_min, n_max + 1, dtype=np.int64)
        pmf_counts = np.zeros((resolutions.size, support.size), dtype=np.int64)
        for j, row in enumerate(n_points_samples):
            pmf_counts[j] = np.bincount(row - n_min, minlength=support.size)
        archive["n_points_support"] = support
        archive["n_points_pmf_counts"] = pmf_counts
        archive["n_points_pmf"] = pmf_counts.astype(float) / cfg.n_realisations
    elif cfg.save_samples:
        archive["length_samples"] = np.stack([r.length_samples for r in results])

    return archive


def _write(cfg: RunConfig, archive: Dict[str, Any], verbose: bool = True) -> None:
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    npz_path = out / f"{cfg.prefix}_pdfs.npz"
    np.savez_compressed(npz_path, **archive)
    cfg.to_json(out / f"{cfg.prefix}_config.json")

    csv_path = out / f"{cfg.prefix}_summary.csv"
    cols = [
        "resolution", "effective_ruler_km", "richardson_dimension",
        "mean", "std", "median", "skewness", "kurtosis",
        "minimum", "p2_5", "q1", "q3", "p97_5", "maximum",
        "n_points_mean", "n_points_std",
    ]
    table = np.column_stack([np.asarray(archive[c], dtype=float) for c in cols])
    np.savetxt(csv_path, table, delimiter=",", header=",".join(cols), comments="", fmt="%.10g")

    if verbose:
        meta = json.loads(str(archive["metadata_json"]))
        print(f"\nwrote {npz_path}")
        print(f"      {csv_path}")
        print(f"      {out / f'{cfg.prefix}_config.json'}")
        if cfg.method == "ys":
            print("      YS archive includes paired (n_p, L) samples for joint resampling")
            if cfg.ys_seeds == "common":
                print(
                    "      common seed stations: each P(L | d) is unchanged, column k of the "
                    "samples is one survey with every ruler (dependent across d)"
                )
        else:
            print("      RP archive includes histogram PDFs P(L | n_p)")
            if cfg.rp_sweep == "nested":
                print(
                    "      nested sweep: each P(L | n_p) is exact, but samples are "
                    "dependent across n_p (same realisations)"
                )
        print(f"elapsed {meta['wall_seconds']:.1f} s")


def main(argv=None) -> None:
    from coastline_config import config_from_args

    try:
        cfg = config_from_args(argv)
    except ValueError as exc:
        # Bad CLI/JSON configuration: avoid a traceback for expected user mistakes.
        print(f"Configuration error:\n  {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    run_campaign(cfg)


if __name__ == "__main__":
    main()
