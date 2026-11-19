"""Monte-Carlo propagation from coastline sampling archives to power-law parameters.

Stage 2 of the pipeline. Reads a ``<prefix>_pdfs.npz`` archive written by
``coastline_run.py`` (archive version >= 6), builds synthetic or stored
length-versus-resolution curves, fits a power law to each one and reports the
distribution of the fitted parameters.

Interpretation
--------------
The sampling archive holds, for every resolution, the distribution P(L | n_p) of the
lengths that the measuring *method* can return. It is not a measurement-error model:
its spread is the variability of the method itself. The distribution of the fitted
parameters is therefore the set of values the method can produce.

Model
-----
With x = n_p - 1 and beta = 1 - 1/D,

    L(n_p) = L0 * x**beta = A * (x / x_c)**beta,      A = L0 * x_c**beta.

``beta`` is the primary fitted exponent; D = 1 / (1 - beta) is derived from it and set
to NaN when beta >= 1 (D undefined). ``L0`` is the model value at n_p = 2 (an
extrapolation far outside the data), which makes it strongly correlated with beta.
``A`` is the model value at the pivot n_p = x_c + 1; the pivot changes only how the
amplitude is reported, never the fitted curve.

Estimators
----------
loglog
    Closed-form (weighted) least squares on ln L = ln A + beta ln(x / x_c).
nonlinear
    Least squares directly on L = A (x / x_c)**beta. For fixed beta the optimal
    amplitude is linear, so the objective is profiled:

        S(beta) = sum w L**2 - (sum w p L)**2 / sum w p**2,   p = (x / x_c)**beta.

    ``profile`` (default): Newton iterations on S(beta), using its analytic first and
    second derivatives, started from the loglog estimate. A curve on which Newton
    fails, does not improve on its starting point, or leaves the --beta-min/--beta-max
    interval is re-solved with the grid method.
    ``profile-grid``: grid + bisection on S'(beta) for every curve (robust, slower).
    ``curve_fit``: scipy.optimize.curve_fit, one call per curve (slow).
    ``--validate-curve-fit N`` re-fits the first N accepted curves with curve_fit (or,
    when curve_fit is the solver, with ``profile``) and reports the differences.
    curve_fit runs with tight tolerances (1e-14), so that the differences measure the
    solvers and not curve_fit's default stopping rule.

Weights
-------
none
    Unit weights.
inverse-variance
    Fixed per resolution, from the archive (``mean``, ``std``):
    nonlinear  w_j = 1 / var(L_j);
    loglog     w_j = 1 / var(ln L_j) ~= mean_j**2 / var(L_j)  (delta method).
    var(L_j) is the spread of the method at resolution j, not a measurement
    uncertainty; these weights favour the resolutions at which the method is most
    reproducible. For YS they use var(L | d) only.

Pivot
-----
All fits use a fixed internal pivot (unweighted geometric mean of x) for
conditioning. The pivot at which A is *reported* is set by ``--pivot``:
``auto``     empirical: chosen after the Monte Carlo so that cov(ln A, beta) = 0 over
             the accepted fits, ln x_c = ln x_int - cov(ln A_int, beta) / var(beta).
             This decorrelates A from beta whatever the weights.
``weights``  geometric mean of x weighted with the log-space-equivalent weights;
             decorrelates A and beta only when the weights match the actual spread of
             the data in the fitted space.
a number     a fixed n_p, so that A is comparable across runs; ``2`` gives A == L0.

Curves (``--curves``, required)
-------------------------------
histogram
    RP only. For every resolution, an independent draw from the saved histogram of
    P(L | n_p), uniform within the selected bin.
samples
    For every resolution, an independent draw (with replacement) of one stored sample:
    a length for RP (needs ``--save-samples`` in coastline_run.py), an (n_p, L) pair
    for YS (which keeps the correlation between n_p and L at that ruler).
realisation
    Every stored realisation is fitted exactly once, in stored order, with the values
    of all resolutions that belong to it. RP: needs ``--rp-sweep nested
    --save-samples``; YS: needs ``--ys-seeds common``. ``--n-sim`` and ``--seed`` are
    ignored.

``histogram`` and ``samples`` give curves whose resolutions are independent ("one
independent campaign per resolution"); ``realisation`` keeps the correlation between
resolutions of one campaign. Since ``samples`` and ``realisation`` use the same stored
values and differ only in how they are paired across resolutions, comparing them
isolates the effect of that correlation; comparing ``histogram`` with ``samples``
isolates the effect of binning.

Every requested (fit, weights) combination is evaluated on the same curves; a curve
is accepted only if every combination produced a valid fit on it. Random numbers are
consumed curve by curve, so results do not depend on the chunk size.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata


MC_ARCHIVE_VERSION = 6
MIN_SOURCE_VERSION = 6      # oldest coastline_run.py archive accepted

FIT_CHOICES = ("loglog", "nonlinear")
WEIGHT_CHOICES = ("none", "inverse-variance")
WEIGHT_TAGS = {"none": "none", "inverse-variance": "ivar"}
SOLVER_CHOICES = ("profile", "profile-grid", "curve_fit")
CURVES_CHOICES = ("histogram", "samples", "realisation")

_INITIAL_HALF_WIDTH = 0.1   # half-width of the first beta grid window
_MAX_EXPANSIONS = 10        # window doubles at most this many times (0.1 -> ~100)
_MAX_BISECTIONS = 200
_NEWTON_MAX_ITERS = 50
_REL_VAR_FLOOR = 1e-12      # floor on (std / mean)**2 to avoid infinite weights
_CURVE_FIT_TOL = 1e-14      # xtol = ftol = gtol of curve_fit
_MIN_ATTEMPTS = 1000        # curves tried before "no valid fit at all" is an error
_MAX_ATTEMPT_FACTOR = 100   # give up after this many attempts per requested curve

_PARAMETERS = ("beta", "Df", "L0", "A", "objective_per_dof")

_CURVES_NOTES = {
    ("rp", "histogram"): "independent piecewise-uniform draw per n_p from the saved "
                         "P(L|n_p) histograms",
    ("rp", "samples"): "independent draw with replacement of one stored length per n_p",
    ("rp", "realisation"): "all stored realisations of a nested archive, fitted exactly "
                           "once in stored order",
    ("ys", "samples"): "independent draw with replacement of one stored (n_p, L) pair "
                       "per ruler",
    ("ys", "realisation"): "all stored common-seed realisations across every ruler, "
                           "fitted exactly once in stored order",
}


# --------------------------------------------------------------------------- archive
def _scalar_string(a: np.ndarray) -> str:
    return str(np.asarray(a).item())


def _source_info(z: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    """Provenance of the sampling archive."""
    meta: Dict[str, Any] = {}
    if "metadata_json" in z.files:
        try:
            meta = json.loads(_scalar_string(z["metadata_json"]))
        except json.JSONDecodeError:
            meta = {}
    version = int(z["archive_version"]) if "archive_version" in z.files else None
    if version is None or version < MIN_SOURCE_VERSION:
        raise ValueError(
            f"sampling archive version {version} is not supported (need >= "
            f"{MIN_SOURCE_VERSION}); regenerate it with coastline_run.py"
        )
    method = _scalar_string(z["method"])
    topology = (_scalar_string(z["topology"]) if "topology" in z.files
                else meta.get("config", {}).get("topology", "open"))
    if topology not in ("open", "closed"):
        raise ValueError(f"unknown topology in archive: {topology!r}")
    if method not in ("rp", "ys"):
        raise ValueError(f"unknown method in archive: {method!r}")
    return {
        "archive_version": version,
        "method": method,
        "topology": topology,
        "rp_sweep": _scalar_string(z["rp_sweep"]),
        "ys_seeds": _scalar_string(z["ys_seeds"]),
        "engine": _scalar_string(z["engine"]),
        "map_max_density_distortion": float(z["map_max_density_distortion"]),
        "metadata": meta,
    }


def _resolution_mask(
    z: np.lib.npyio.NpzFile,
    method: str,
    np_min: float | None,
    np_max: float | None,
) -> NDArray[np.bool_]:
    proxy = np.asarray(z["resolution"] if method == "rp" else z["n_points_mean"], dtype=float)
    mask = np.ones(proxy.size, dtype=bool)
    if np_min is not None:
        mask &= proxy >= np_min
    if np_max is not None:
        mask &= proxy <= np_max
    if mask.sum() < 3:
        raise ValueError(
            "fewer than 3 resolutions survive the n_p range; broaden --np-min/--np-max"
        )
    return mask


# --------------------------------------------------------------------------- sampling
# The random samplers consume their random numbers row by row (curve by curve), so
# that drawing m1 curves and then m2 curves gives exactly the same curves as drawing
# m1 + m2 at once: results do not depend on the chunk size. Every sampler returns
# (n_points, lengths), each of shape (m, n_res); n_points is None when it is fixed
# by the resolution (RP).
class _RPHistogramSampler:
    """RP: independent histogram draws, preserving explicitly flagged point masses."""

    def __init__(self, hist_counts: NDArray[np.int64], edges: NDArray[np.float64],
                 is_pointmass: NDArray[np.bool_] | None = None,
                 pointmass: NDArray[np.float64] | None = None) -> None:
        counts = hist_counts.astype(float)
        total = counts.sum(axis=1)
        if np.any(total <= 0):
            j = int(np.flatnonzero(total <= 0)[0])
            raise ValueError(f"empty RP histogram at selected resolution index {j}")
        self.cdf = np.cumsum(counts, axis=1) / total[:, None]
        self.edges = edges
        self.n_res, self.n_bins = counts.shape
        self.is_pointmass = (np.zeros(self.n_res, dtype=bool) if is_pointmass is None
                             else np.asarray(is_pointmass, dtype=bool))
        self.pointmass = (np.zeros(self.n_res) if pointmass is None
                          else np.asarray(pointmass, dtype=float))

    def __call__(self, rng: np.random.Generator, m: int):
        U = rng.random((m, self.n_res, 2))
        out = np.empty((m, self.n_res))
        for j in range(self.n_res):
            b = np.minimum(np.searchsorted(self.cdf[j], U[:, j, 0], side="right"), self.n_bins - 1)
            lo = self.edges[j, b]
            out[:, j] = (self.pointmass[j] if self.is_pointmass[j] else
                         lo + U[:, j, 1] * (self.edges[j, b + 1] - lo))
        return None, out


class _BootstrapSampler:
    """One stored sample per resolution, drawn with replacement, independently.

    RP: ``n_points`` is None and only lengths are drawn. YS: the (n_p, L) pair of
    one stored survey is drawn, which keeps the correlation between n_p and L.
    """

    def __init__(self, n_points: NDArray[np.int64] | None, lengths: NDArray[np.float64]) -> None:
        self.n_points = n_points                 # (n_res, n_samples) or None
        self.lengths = lengths                   # (n_res, n_samples)
        self.cols = np.arange(lengths.shape[0])[None, :]

    def __call__(self, rng: np.random.Generator, m: int):
        idx = rng.integers(0, self.lengths.shape[1], size=(m, self.lengths.shape[0]))
        n_points = None if self.n_points is None else self.n_points[self.cols, idx].astype(float)
        return n_points, self.lengths[self.cols, idx]


class _RealisationSampler:
    """Deterministic traversal of whole stored realisations (no random numbers)."""

    def __init__(self, n_points: NDArray[np.int64] | None, lengths: NDArray[np.float64]) -> None:
        self.n_points = n_points                 # (n_res, n_real) or None
        self.lengths = lengths
        self.n_real = lengths.shape[1]

    def slice(self, start: int, stop: int):
        n_points = (None if self.n_points is None
                    else self.n_points[:, start:stop].T.astype(float))
        return n_points, self.lengths[:, start:stop].T


@dataclass
class _Source:
    """Everything the fits need from the sampling archive."""

    path: Path
    method: str
    info: Dict[str, Any]
    resolution: NDArray[np.float64]
    n_points_mean: NDArray[np.float64]
    length_mean: NDArray[np.float64]
    length_std: NDArray[np.float64]
    n_fixed: NDArray[np.float64] | None
    sampler: Any
    n_real_stored: int | None
    notes: List[str] = field(default_factory=list)


def _load_source(
    path: Path, curves: str, np_min: float | None, np_max: float | None
) -> _Source:
    with np.load(path, allow_pickle=False) as z:
        info = _source_info(z)
        method = info["method"]
        mask = _resolution_mask(z, method, np_min, np_max)
        has_samples = "length_samples" in z.files
        notes: List[str] = []

        if method == "rp":
            if curves == "histogram":
                sampler = _RPHistogramSampler(
                    np.asarray(z["length_hist_counts"], dtype=np.int64)[mask],
                    np.asarray(z["length_bin_edges"], dtype=float)[mask],
                    (np.asarray(z["hist_is_pointmass"], dtype=bool)[mask]
                     if "hist_is_pointmass" in z.files else None),
                    (np.asarray(z["hist_pointmass"], dtype=float)[mask]
                     if "hist_pointmass" in z.files else None),
                )
            else:
                if not has_samples:
                    raise ValueError(
                        f"--curves {curves} on RP needs the raw samples: regenerate the "
                        "archive with coastline_run.py --save-samples, or use --curves histogram"
                    )
                if curves == "realisation" and info["rp_sweep"] != "nested":
                    raise ValueError(
                        "--curves realisation on RP needs a nested archive "
                        "(coastline_run.py --rp-sweep nested --save-samples)"
                    )
                samples = np.asarray(z["length_samples"], dtype=float)[mask]
                sampler = (_RealisationSampler(None, samples) if curves == "realisation"
                           else _BootstrapSampler(None, samples))
            n_fixed = np.asarray(z["resolution"], dtype=float)[mask]
            if info["rp_sweep"] == "nested" and curves != "realisation":
                notes.append(
                    "nested archive with independent curves: every P(L|n_p) is exact, and "
                    "each n_p is drawn independently (one independent campaign per n_p)"
                )
        else:
            if curves == "histogram":
                raise ValueError(
                    "--curves histogram is not available for YS: n_p varies from survey to "
                    "survey and the histograms do not keep its pairing with L; use "
                    "--curves samples or realisation"
                )
            if curves == "realisation" and info["ys_seeds"] != "common":
                raise ValueError(
                    "--curves realisation on YS needs an archive with common seed stations "
                    "(coastline_run.py --method ys --ys-seeds common)"
                )
            ys_n = np.asarray(z["n_points_samples"], dtype=np.int64)[mask]
            ys_l = np.asarray(z["length_samples"], dtype=float)[mask]
            if ys_n.shape != ys_l.shape:
                raise ValueError("YS paired sample arrays have inconsistent shapes")
            sampler = (_RealisationSampler(ys_n, ys_l) if curves == "realisation"
                       else _BootstrapSampler(ys_n, ys_l))
            n_fixed = None
            if info["ys_seeds"] == "common" and curves == "samples":
                notes.append(
                    "common-seed archive with --curves samples: every P(L|d) is exact, and "
                    "each ruler is drawn independently (one independent survey per ruler)"
                )

        n_real_stored = None
        if curves == "realisation":
            n_real_stored = sampler.n_real
            n_meta = int(z["n_realisations"])
            if n_meta != n_real_stored:
                notes.append(
                    f"archive metadata says n_realisations={n_meta}, but the stored sample "
                    f"arrays contain {n_real_stored}; using the stored arrays"
                )

        return _Source(
            path=path,
            method=method,
            info=info,
            resolution=np.asarray(z["resolution"], dtype=float)[mask],
            n_points_mean=np.asarray(z["n_points_mean"], dtype=float)[mask],
            length_mean=np.asarray(z["mean"], dtype=float)[mask],
            length_std=np.asarray(z["std"], dtype=float)[mask],
            n_fixed=n_fixed,
            sampler=sampler,
            n_real_stored=n_real_stored,
            notes=notes,
        )


# --------------------------------------------------------------------------- weights
def _weights(
    scheme: str, fit: str, mean: NDArray[np.float64], std: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-resolution weights; identical for every synthetic curve."""
    if scheme == "none":
        return np.ones(mean.size, dtype=float)
    if np.any(mean <= 0.0) or not np.all(np.isfinite(std)):
        raise ValueError("inverse-variance weights require positive means and finite stds")
    rel_var = np.maximum((std / mean) ** 2, _REL_VAR_FLOOR)
    if fit == "loglog":
        return 1.0 / rel_var
    return 1.0 / (rel_var * mean * mean)


def _log_equivalent_weights(
    fit: str, w: NDArray[np.float64], mean: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Weights expressed in log space (used by the 'weights' pivot)."""
    return w if fit == "loglog" else w * mean * mean


def _n_eff(w: NDArray[np.float64]) -> float:
    """Effective number of resolutions, (sum w)^2 / sum w^2 (Kish)."""
    return float(w.sum() ** 2 / np.sum(w * w))


# --------------------------------------------------------------------------- fits
def _fit_loglog(
    u: NDArray[np.float64],
    y: NDArray[np.float64],
    w: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Weighted linear fit y = lnA + beta*u, row-wise. u, y: (m, n); w: (n,)."""
    wsum = w.sum()
    ubar = (u * w).sum(axis=1) / wsum
    ybar = (y * w).sum(axis=1) / wsum
    du = u - ubar[:, None]
    sxx = (w * du * du).sum(axis=1)
    good = sxx > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (w * du * (y - ybar[:, None])).sum(axis=1) / sxx
    ln_a = ybar - beta * ubar
    resid = y - ln_a[:, None] - beta[:, None] * u
    obj = (w * resid * resid).sum(axis=1)
    return beta, ln_a, obj, good


def _profile_terms(
    beta: NDArray[np.float64],
    u: NDArray[np.float64],
    L: NDArray[np.float64],
    w: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Profiled amplitude, objective and d(objective)/d(beta) at the given betas."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        p = np.exp(beta[:, None] * u)
        wp = w * p
        amp = (wp * L).sum(axis=1) / (wp * p).sum(axis=1)
        r = L - amp[:, None] * p
        obj = (w * r * r).sum(axis=1)
        grad = -2.0 * amp * (wp * u * r).sum(axis=1)
    return amp, obj, grad


def _fit_profile(
    u: NDArray[np.float64],
    L: NDArray[np.float64],
    w: NDArray[np.float64],
    beta0: NDArray[np.float64],
    beta_min: float | None,
    beta_max: float | None,
    n_grid: int,
    xtol: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Grid + bisection profile fit of L = A exp(beta u) (robust; used as fallback)."""
    m = L.shape[0]
    lo_bound = -np.inf if beta_min is None else float(beta_min)
    hi_bound = np.inf if beta_max is None else float(beta_max)

    center = np.where(np.isfinite(beta0), beta0, 0.0)
    center = np.clip(center, lo_bound, hi_bound)
    half = np.full(m, _INITIAL_HALF_WIDTH)
    bracket_lo = np.full(m, np.nan)
    bracket_hi = np.full(m, np.nan)
    bracketed = np.zeros(m, dtype=bool)
    frac = np.linspace(0.0, 1.0, n_grid)

    todo = np.arange(m)
    for _ in range(_MAX_EXPANSIONS + 1):
        if todo.size == 0:
            break
        uu, LL = u[todo], L[todo]
        g_lo = np.maximum(center[todo] - half[todo], lo_bound)
        g_hi = np.minimum(center[todo] + half[todo], hi_bound)

        best = np.full(todo.size, np.inf)
        k_best = np.zeros(todo.size, dtype=np.int64)
        for k, f in enumerate(frac):
            _, obj, _ = _profile_terms(g_lo + f * (g_hi - g_lo), uu, LL, w)
            better = obj < best
            best[better] = obj[better]
            k_best[better] = k

        step = (g_hi - g_lo) / (n_grid - 1)
        at_left = (k_best == 0) & (g_lo > lo_bound)
        at_right = (k_best == n_grid - 1) & (g_hi < hi_bound)
        on_edge = at_left | at_right | ~np.isfinite(best)

        ok = ~on_edge
        rows = todo[ok]
        kb = k_best[ok]
        bracket_lo[rows] = np.maximum(g_lo[ok] + (kb - 1) * step[ok], g_lo[ok])
        bracket_hi[rows] = np.minimum(g_lo[ok] + (kb + 1) * step[ok], g_hi[ok])
        bracketed[rows] = True

        rows = todo[on_edge]
        center[rows] = np.where(at_right[on_edge], g_hi[on_edge], g_lo[on_edge])
        half[rows] *= 2.0
        todo = rows

    beta = np.full(m, np.nan)
    ln_a = np.full(m, np.nan)
    obj_out = np.full(m, np.nan)
    at_bound = np.zeros(m, dtype=bool)

    idx = np.flatnonzero(bracketed)
    if idx.size:
        uu, LL = u[idx], L[idx]
        a, b = bracket_lo[idx], bracket_hi[idx]
        eps = np.finfo(float).eps
        for _ in range(_MAX_BISECTIONS):
            c = 0.5 * (a + b)
            tol = np.maximum(xtol, 8.0 * eps * np.maximum(1.0, np.abs(c)))
            if np.all(b - a <= tol):
                break
            _, _, grad = _profile_terms(c, uu, LL, w)
            min_is_left = grad > 0.0
            b = np.where(min_is_left, c, b)
            a = np.where(min_is_left, a, c)
        bb = 0.5 * (a + b)
        amp, obj, _ = _profile_terms(bb, uu, LL, w)
        beta[idx] = bb
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_a[idx] = np.log(amp)
        obj_out[idx] = obj
        tol = np.maximum(xtol, 8.0 * eps * np.maximum(1.0, np.abs(bb)))
        at_bound[idx] = (np.abs(bb - lo_bound) <= tol) | (np.abs(bb - hi_bound) <= tol)

    good = bracketed & np.isfinite(beta) & np.isfinite(ln_a) & np.isfinite(obj_out)
    return beta, ln_a, obj_out, good, at_bound


def _profile_sums(beta, u, L, w):
    """The six sums that give S, S' and S'' of the profiled objective, in one pass."""
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.exp(beta[:, None] * u)
        wp = w * p
        wpu = wp * u
        a0 = (wp * L).sum(axis=1)
        a1 = (wpu * L).sum(axis=1)
        a2 = (wpu * u * L).sum(axis=1)
        b0 = (wp * p).sum(axis=1)
        b1 = 2.0 * (wpu * p).sum(axis=1)
        b2 = 4.0 * (wpu * u * p).sum(axis=1)
    return a0, a1, a2, b0, b1, b2


def _fit_profile_newton(
    u: NDArray[np.float64],
    L: NDArray[np.float64],
    w: NDArray[np.float64],
    beta0: NDArray[np.float64],
    beta_min: float | None,
    beta_max: float | None,
    n_grid: int,
    xtol: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Newton on the profiled objective, with the grid method as fallback.

    With a = sum w p L and b = sum w p^2 (primes: derivatives in beta),
        S   = c - a^2 / b
        S'  = -2 a a'/b + a^2 b'/b^2
        S'' = -2 (a'^2 + a a'')/b + 4 a a' b'/b^2 + a^2 b''/b^2 - 2 a^2 b'^2/b^3.
    """
    m = L.shape[0]
    eps = np.finfo(float).eps
    beta = np.where(np.isfinite(beta0), beta0, 0.0).astype(float)
    c = (w * L * L).sum(axis=1)
    _, obj_start, _ = _profile_terms(beta, u, L, w)

    done = np.zeros(m, dtype=bool)
    fail = ~np.isfinite(obj_start)
    for _ in range(_NEWTON_MAX_ITERS):
        act = np.flatnonzero(~done & ~fail)
        if act.size == 0:
            break
        a0, a1, a2, b0, b1, b2 = _profile_sums(beta[act], u[act], L[act], w)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            g = -2.0 * a0 * a1 / b0 + a0 * a0 * b1 / (b0 * b0)
            h = (-2.0 * (a1 * a1 + a0 * a2) / b0 + 4.0 * a0 * a1 * b1 / (b0 * b0)
                 + a0 * a0 * b2 / (b0 * b0) - 2.0 * a0 * a0 * b1 * b1 / (b0 * b0 * b0))
            step = -g / h
        bad = ~(h > 0.0) | ~np.isfinite(step)
        fail[act[bad]] = True
        go = act[~bad]
        beta[go] += step[~bad]
        tol = np.maximum(xtol, 8.0 * eps * np.maximum(1.0, np.abs(beta[go])))
        done[go] |= np.abs(step[~bad]) <= tol

    amp, obj, _ = _profile_terms(beta, u, L, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_a = np.log(amp)
    ok = done & ~fail & np.isfinite(ln_a) & np.isfinite(obj)
    ok &= obj <= obj_start * (1.0 + 1e-12) + 1e-300      # never worse than the start
    if beta_min is not None:
        ok &= beta > beta_min
    if beta_max is not None:
        ok &= beta < beta_max
    at_bound = np.zeros(m, dtype=bool)

    redo = np.flatnonzero(~ok)
    if redo.size:
        rb, rl, ro, rg, rab = _fit_profile(
            u[redo], L[redo], w, beta0[redo], beta_min, beta_max, n_grid, xtol
        )
        beta[redo], ln_a[redo], obj[redo], at_bound[redo] = rb, rl, ro, rab
        ok[redo] = rg
    good = ok & np.isfinite(beta) & np.isfinite(ln_a) & np.isfinite(obj)
    return beta, ln_a, obj, good, at_bound


def _power_model(u: NDArray[np.float64], amp: float, beta: float) -> NDArray[np.float64]:
    return amp * np.exp(beta * u)


def _power_model_jac(u: NDArray[np.float64], amp: float, beta: float) -> NDArray[np.float64]:
    p = np.exp(beta * u)
    return np.column_stack([p, amp * u * p])


def _fit_curve_fit(
    u: NDArray[np.float64],
    L: NDArray[np.float64],
    w: NDArray[np.float64],
    weighted: bool,
    beta0: NDArray[np.float64],
    ln_a0: NDArray[np.float64],
    beta_min: float | None,
    beta_max: float | None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_]]:
    """Per-curve scipy.optimize.curve_fit of L = A exp(beta u)."""
    from scipy.optimize import OptimizeWarning, curve_fit

    m = L.shape[0]
    beta = np.full(m, np.nan)
    ln_a = np.full(m, np.nan)
    obj = np.full(m, np.nan)
    at_bound = np.zeros(m, dtype=bool)

    sigma = 1.0 / np.sqrt(w) if weighted else None
    lo_bound = -np.inf if beta_min is None else float(beta_min)
    hi_bound = np.inf if beta_max is None else float(beta_max)
    bounded = beta_min is not None or beta_max is not None
    kwargs = (
        {"bounds": ([-np.inf, lo_bound], [np.inf, hi_bound]), "method": "trf"}
        if bounded
        else {"method": "lm"}
    )
    kwargs.update(xtol=_CURVE_FIT_TOL, ftol=_CURVE_FIT_TOL, gtol=_CURVE_FIT_TOL)
    nudge = 1e-9

    for i in range(m):
        b_start = beta0[i] if np.isfinite(beta0[i]) else 0.0
        if bounded:
            b_start = min(max(b_start, lo_bound + nudge), hi_bound - nudge)
        a_start = np.exp(ln_a0[i]) if np.isfinite(ln_a0[i]) else float(np.mean(L[i]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                popt, _ = curve_fit(
                    _power_model, u[i], L[i], p0=[a_start, b_start],
                    sigma=sigma, jac=_power_model_jac, **kwargs,
                )
            except (RuntimeError, ValueError):
                continue
        amp, bi = float(popt[0]), float(popt[1])
        if not (np.isfinite(amp) and np.isfinite(bi) and amp > 0.0):
            continue
        r = L[i] - amp * np.exp(bi * u[i])
        beta[i] = bi
        ln_a[i] = np.log(amp)
        obj[i] = float(np.sum(w * r * r))
        tol = 1e-8 * max(1.0, abs(bi))
        at_bound[i] = abs(bi - lo_bound) <= tol or abs(bi - hi_bound) <= tol

    good = np.isfinite(beta) & np.isfinite(ln_a) & np.isfinite(obj)
    return beta, ln_a, obj, good, at_bound


def _fit_nonlinear(solver, u, L, w, weighted, beta0, ln_a0, beta_min, beta_max, n_grid, xtol):
    if solver == "profile":
        return _fit_profile_newton(u, L, w, beta0, beta_min, beta_max, n_grid, xtol)
    if solver == "profile-grid":
        return _fit_profile(u, L, w, beta0, beta_min, beta_max, n_grid, xtol)
    if solver == "curve_fit":
        return _fit_curve_fit(u, L, w, weighted, beta0, ln_a0, beta_min, beta_max)
    raise ValueError(f"unknown nonlinear solver {solver!r}")


def _fit_all(
    combos: List[Tuple[str, str]],
    u: NDArray[np.float64],
    L: NDArray[np.float64],
    weights: Dict[Tuple[str, str], NDArray[np.float64]],
    solver: str,
    beta_min: float | None,
    beta_max: float | None,
    beta_grid: int,
    beta_xtol: float,
) -> Dict[Tuple[str, str], Dict[str, NDArray]]:
    """Run every combination on the same curves. u, L: (m, n)."""
    y = np.log(L)
    schemes = list(dict.fromkeys(s for _, s in combos))
    loglog = {s: _fit_loglog(u, y, weights[("loglog", s)]) for s in schemes}

    out: Dict[Tuple[str, str], Dict[str, NDArray]] = {}
    for fit, scheme in combos:
        if fit == "loglog":
            beta, ln_a, obj, good = loglog[scheme]
            at_bound = np.zeros(L.shape[0], dtype=bool)
        else:
            beta, ln_a, obj, good, at_bound = _fit_nonlinear(
                solver, u, L, weights[("nonlinear", scheme)], scheme != "none",
                loglog[scheme][0], loglog[scheme][1], beta_min, beta_max, beta_grid, beta_xtol,
            )
        good = good & np.isfinite(beta) & np.isfinite(ln_a) & np.isfinite(obj)
        out[(fit, scheme)] = {
            "beta": beta, "ln_a": ln_a, "obj": obj, "good": good, "at_bound": at_bound,
            "beta0": loglog[scheme][0], "ln_a0": loglog[scheme][1],
        }
    return out


# --------------------------------------------------------------------------- statistics
def _summary(x: NDArray[np.float64]) -> Dict[str, float]:
    f = x[np.isfinite(x)]
    keys = ("mean", "std", "median", "p2_5", "p16", "p84", "p97_5")
    if f.size < 2:
        return {"n_finite": int(f.size), **{k: float("nan") for k in keys}}
    q = np.percentile(f, [50.0, 2.5, 16.0, 84.0, 97.5])
    return {
        "n_finite": int(f.size),
        "mean": float(np.mean(f)),
        "std": float(np.std(f, ddof=1)),
        "median": float(q[0]),
        "p2_5": float(q[1]),
        "p16": float(q[2]),
        "p84": float(q[3]),
        "p97_5": float(q[4]),
    }


def _hist_range(f: NDArray[np.float64], tail: float) -> Tuple[float, float]:
    """Histogram range: [tail, 1 - tail] quantiles (full range when tail == 0)."""
    lo, hi = (np.quantile(f, [tail, 1.0 - tail]) if tail > 0.0 else (f.min(), f.max()))
    if not hi > lo:
        lo, hi = f.min(), f.max()
    # A range of a few ulps (a deterministic input, e.g. a point-mass P(L | d)) cannot
    # be cut into bins any more than an empty one: widen it around its centre.
    # The artificial plotting range must scale with the parameter, rather than
    # with 1: an almost-zero exponent should not acquire a visibly wide PDF.
    # The 1e-12 floor only sets a harmless display scale for exactly zero.
    scale = max(abs(lo), abs(hi), 1e-12)
    if not hi - lo > 1e-12 * scale:
        mid = 0.5 * (lo + hi)
        pad = 0.5e-9 * scale
        lo, hi = mid - pad, mid + pad
    return float(lo), float(hi)


def _pdf1d(x: NDArray[np.float64], bins: int, tail: float):
    """Counts, density, edges, underflow, overflow. Density is normalised by the number
    of finite samples, so it integrates to the in-range fraction."""
    f = x[np.isfinite(x)]
    if f.size == 0:
        return None
    lo, hi = _hist_range(f, tail)
    counts, edges = np.histogram(f, bins=bins, range=(lo, hi))
    density = counts / (f.size * np.diff(edges))
    under = int(np.count_nonzero(f < lo))
    over = int(np.count_nonzero(f > hi))
    return counts.astype(np.int64), density.astype(float), edges.astype(float), under, over


def _pdf2d(x: NDArray[np.float64], y: NDArray[np.float64], bins: int, tail: float):
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return None
    xs, ys = x[keep], y[keep]
    rx, ry = _hist_range(xs, tail), _hist_range(ys, tail)
    counts, xe, ye = np.histogram2d(xs, ys, bins=bins, range=(rx, ry))
    area = np.diff(xe)[:, None] * np.diff(ye)[None, :]
    density = counts / (keep.sum() * area)
    outside = int(keep.sum() - counts.sum())
    return counts.astype(np.int64), density, xe, ye, outside


def _corr(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    xs, ys = x[keep], y[keep]
    if np.std(xs) == 0.0 or np.std(ys) == 0.0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def _spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Rank correlation (robust to the heavy tails of L0 and Df)."""
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    # Average ranks for ties (including constant arrays), as in Spearman's rho.
    rx = rankdata(x[keep], method="average")
    ry = rankdata(y[keep], method="average")
    return _corr(rx, ry)


def _derived(
    beta: NDArray[np.float64],
    ln_a_int: NDArray[np.float64],
    ln_xc_int: float,
    ln_xc_rep: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(D, L0, A_reported) from the exponent and the internal-pivot amplitude."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        Df = np.where(beta < 1.0, 1.0 / (1.0 - beta), np.nan)
        L0 = np.exp(ln_a_int - beta * ln_xc_int)
        A = np.exp(ln_a_int + beta * (ln_xc_rep - ln_xc_int))
    return Df, L0, A


def _empirical_pivot(
    beta: NDArray[np.float64], ln_a_int: NDArray[np.float64], ln_xc_int: float
) -> float:
    """ln x_c at which cov(ln A, beta) = 0 over the fits (internal pivot if undefined)."""
    keep = np.isfinite(beta) & np.isfinite(ln_a_int)
    if keep.sum() < 3:
        return ln_xc_int
    b, la = beta[keep], ln_a_int[keep]
    var_b = np.var(b)
    if not var_b > 0.0:
        return ln_xc_int
    cov = np.mean((la - la.mean()) * (b - b.mean()))
    return float(ln_xc_int - cov / var_b)


# --------------------------------------------------------------------------- curve loop
def _combo_tag(fit: str, scheme: str) -> str:
    return f"{fit}_{WEIGHT_TAGS[scheme]}"


def _fit_curves(
    src: _Source,
    curves: str,
    combos: List[Tuple[str, str]],
    n_sim: int,
    seed: int,
    rows_per_chunk: int,
    ln_xc_int: float,
    nonlinear_solver: str,
    validate_curve_fit: int,
    fit_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the curves chunk by chunk and fit every combination on them.

    histogram/samples: draw curves until ``n_sim`` are accepted.
    realisation: traverse every stored realisation once, in stored order.
    """
    weights = fit_kwargs["weights"]
    nonlinear_combos = [c for c in combos if c[0] == "nonlinear"]
    other_solver = "curve_fit" if nonlinear_solver != "curve_fit" else "profile"
    realisation = curves == "realisation"
    n_target = int(src.n_real_stored) if realisation else n_sim
    rng = None if realisation else np.random.default_rng(seed)

    store = {
        c: {k: np.empty(n_target) for k in ("beta", "ln_a", "obj")}
        | {"at_bound": np.zeros(n_target, bool)}
        for c in combos
    }
    source_indices = np.empty(n_target, dtype=np.int64) if realisation else None
    validation = {
        c: {"n": 0, "n_fail": 0, "max_abs_dbeta": 0.0, "max_rel_dA": 0.0}
        for c in nonlinear_combos
    }
    n_valid = n_attempted = n_rejected_data = n_rejected_fit = 0
    source_pos = 0

    while (source_pos < n_target) if realisation else (n_valid < n_target):
        if realisation:
            start, stop = source_pos, min(source_pos + rows_per_chunk, n_target)
            n_points, lengths = src.sampler.slice(start, stop)
            source_pos = stop
            m = stop - start
        else:
            # Abort only on a global failure: a short final chunk may legitimately
            # contain no acceptable curve at all.
            if n_valid == 0 and n_attempted >= max(_MIN_ATTEMPTS, n_target):
                raise RuntimeError(
                    f"no valid fit in the first {n_attempted} curves; check the archive, "
                    "the n_p range and the beta bounds"
                )
            if n_attempted > _MAX_ATTEMPT_FACTOR * n_target:
                raise RuntimeError(
                    f"acceptance rate too low: {n_valid} valid fits in {n_attempted} curves"
                )
            m = min(rows_per_chunk, n_target - n_valid)
            n_points, lengths = src.sampler(rng, m)
            start = -1

        if n_points is None:
            n_points = np.broadcast_to(src.n_fixed[None, :], lengths.shape)
        n_attempted += m

        x = n_points - 1.0
        data_ok = np.all(
            (x > 0.0) & np.isfinite(x) & (lengths > 0.0) & np.isfinite(lengths), axis=1
        )
        rows = np.flatnonzero(data_ok)
        n_rejected_data += m - rows.size
        if rows.size == 0:
            continue

        L = lengths[rows]
        u = np.log(x[rows]) - ln_xc_int
        res = _fit_all(combos, u, L, solver=nonlinear_solver, **fit_kwargs)

        row_ok = np.ones(rows.size, dtype=bool)
        for c in combos:
            row_ok &= res[c]["good"]
        n_rejected_fit += int(rows.size - row_ok.sum())
        acc = np.flatnonzero(row_ok)
        if acc.size == 0:
            continue

        take = acc.size if realisation else min(acc.size, n_target - n_valid)
        acc = acc[:take]
        sl = slice(n_valid, n_valid + take)
        for c in combos:
            for k in ("beta", "ln_a", "obj", "at_bound"):
                store[c][k][sl] = res[c][k][acc]
        if realisation:
            source_indices[sl] = start + rows[acc]

        for c in nonlinear_combos:
            need = validate_curve_fit - validation[c]["n"]
            if need <= 0:
                continue
            sel = acc[:need]
            ob, ol, _, og, _ = _fit_nonlinear(
                other_solver, u[sel], L[sel], weights[c], c[1] != "none",
                res[c]["beta0"][sel], res[c]["ln_a0"][sel],
                fit_kwargs["beta_min"], fit_kwargs["beta_max"],
                fit_kwargs["beta_grid"], fit_kwargs["beta_xtol"],
            )
            v = validation[c]
            v["n"] += int(sel.size)
            v["n_fail"] += int((~og).sum())
            if og.any():
                db = np.abs(ob[og] - res[c]["beta"][sel][og])
                da = np.abs(np.expm1(ol[og] - res[c]["ln_a"][sel][og]))
                v["max_abs_dbeta"] = max(v["max_abs_dbeta"], float(db.max()))
                v["max_rel_dA"] = max(v["max_rel_dA"], float(da.max()))

        n_valid += take

    if n_valid == 0:
        raise RuntimeError("no valid curves/fits were produced")
    if realisation:
        for c in combos:
            for k in ("beta", "ln_a", "obj", "at_bound"):
                store[c][k] = store[c][k][:n_valid]
        source_indices = source_indices[:n_valid]

    return {
        "store": store, "validation": validation, "other_solver": other_solver,
        "n_target": n_target, "n_valid": n_valid, "n_attempted": n_attempted,
        "n_rejected_data": n_rejected_data, "n_rejected_fit": n_rejected_fit,
        "source_indices": source_indices,
    }


# --------------------------------------------------------------------------- driver
def default_prefix(input_npz: str | Path, curves: str) -> str:
    """``<archive prefix>_<curves>``: runs of different modes never overwrite each other."""
    stem = Path(input_npz).name
    for suffix in (".npz", "_pdfs"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return f"{stem}_{curves}"


def _check_arguments(
    curves, n_sim, output_bins, joint_bins, pivot, nonlinear_solver, validate_curve_fit,
    beta_min, beta_max, beta_grid, beta_xtol, chunk_elements, chunk_sim, hist_tail,
) -> float | None:
    """Validate the arguments; return the user pivot n_p (None for auto/weights)."""
    if curves not in CURVES_CHOICES:
        raise ValueError(f"curves must be one of {CURVES_CHOICES}")
    if curves != "realisation" and n_sim < 2:
        raise ValueError("n_sim must be >= 2")
    if chunk_sim is not None and chunk_sim < 1:
        raise ValueError("chunk_sim must be >= 1")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be >= 1")
    if output_bins < 10 or joint_bins < 10:
        raise ValueError("histogram bin counts must be >= 10")
    if beta_grid < 3:
        raise ValueError("beta_grid must be >= 3")
    if beta_xtol <= 0.0:
        raise ValueError("beta_xtol must be positive")
    if validate_curve_fit < 0:
        raise ValueError("validate_curve_fit must be >= 0")
    if beta_min is not None and beta_max is not None and not beta_min < beta_max:
        raise ValueError("require beta_min < beta_max")
    if nonlinear_solver not in SOLVER_CHOICES:
        raise ValueError(f"unknown nonlinear solver {nonlinear_solver!r}")
    if not 0.0 <= hist_tail < 0.5:
        raise ValueError("hist_tail must be in [0, 0.5)")
    if pivot in ("auto", "weights"):
        return None
    pivot_np = float(pivot)
    if not pivot_np > 1.0:
        raise ValueError("pivot must be 'auto', 'weights' or a number > 1 (an n_p value)")
    return pivot_np


def run_mc(
    input_npz: str | Path,
    curves: str,
    n_sim: int = 100_000,
    seed: int = 20260814,
    np_min: float | None = None,
    np_max: float | None = None,
    output_bins: int = 100,
    joint_bins: int = 80,
    output_dir: str | Path = "results/mc_fit",
    prefix: str | None = None,
    fits: List[str] = ("loglog",),
    weight_schemes: List[str] = ("none",),
    pivot: str = "auto",
    nonlinear_solver: str = "profile",
    validate_curve_fit: int = 0,
    beta_min: float | None = None,
    beta_max: float | None = None,
    beta_grid: int = 32,
    beta_xtol: float = 1e-10,
    chunk_elements: int = 1_048_576,
    chunk_sim: int | None = None,
    hist_tail: float = 1e-3,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    pivot_np_user = _check_arguments(
        curves, n_sim, output_bins, joint_bins, pivot, nonlinear_solver, validate_curve_fit,
        beta_min, beta_max, beta_grid, beta_xtol, chunk_elements, chunk_sim, hist_tail,
    )
    fits = list(dict.fromkeys(fits))
    weight_schemes = list(dict.fromkeys(weight_schemes))
    combos = [(f, s) for f in fits for s in weight_schemes]
    prefix = prefix or default_prefix(input_npz, curves)

    t0 = time.perf_counter()
    src = _load_source(Path(input_npz), curves, np_min, np_max)
    notes = list(src.notes)
    if curves == "realisation":
        notes.append("--curves realisation fits every stored realisation once; "
                     "--n-sim and --seed are ignored")
    if verbose:
        for msg in notes:
            print(f"note: {msg}")

    n_res = src.resolution.size
    x_mean = src.n_points_mean - 1.0
    if np.any(x_mean <= 0.0):
        raise ValueError("every fitted resolution needs mean n_p > 1")
    ln_x_mean = np.log(x_mean)
    ln_xc_int = float(np.mean(ln_x_mean))

    weights: Dict[Tuple[str, str], NDArray[np.float64]] = {
        (fit, scheme): _weights(scheme, fit, src.length_mean, src.length_std)
        for scheme in weight_schemes for fit in FIT_CHOICES
    }
    fit_kwargs = dict(
        weights=weights, beta_min=beta_min, beta_max=beta_max,
        beta_grid=beta_grid, beta_xtol=beta_xtol,
    )
    rows_per_chunk = chunk_sim if chunk_sim is not None else max(1, chunk_elements // n_res)
    run = _fit_curves(
        src, curves, combos, n_sim, seed, rows_per_chunk, ln_xc_int,
        nonlinear_solver, validate_curve_fit, fit_kwargs,
    )
    store, validation = run["store"], run["validation"]

    # ---- fit of the mean curve (a point estimate, for reference) -------------------
    mean_curve = _fit_all(
        combos, (ln_x_mean - ln_xc_int)[None, :], src.length_mean[None, :],
        solver=nonlinear_solver, **fit_kwargs,
    )

    # ---- reporting pivots --------------------------------------------------------
    ln_xc_rep: Dict[Tuple[str, str], float] = {}
    for c in combos:
        if pivot == "auto":
            ln_xc_rep[c] = _empirical_pivot(store[c]["beta"], store[c]["ln_a"], ln_xc_int)
        elif pivot == "weights":
            wl = _log_equivalent_weights(c[0], weights[c], src.length_mean)
            ln_xc_rep[c] = float(np.sum(wl * ln_x_mean) / np.sum(wl))
        else:
            ln_xc_rep[c] = float(np.log(pivot_np_user - 1.0))
        pnp = float(np.exp(ln_xc_rep[c]) + 1.0)
        if not src.n_points_mean.min() <= pnp <= src.n_points_mean.max():
            msg = (
                f"[{_combo_tag(*c)}] pivot n_p = {pnp:.6g} lies outside the fitted n_p range "
                f"[{src.n_points_mean.min():g}, {src.n_points_mean.max():g}]; A is an "
                "extrapolation"
            )
            notes.append(msg)
            if verbose:
                print(f"note: {msg}")

    # ---- reduction ---------------------------------------------------------------
    dof = n_res - 2
    archive: Dict[str, np.ndarray] = {
        "mc_archive_version": np.int64(MC_ARCHIVE_VERSION),
        "method": np.array(src.method),
        "topology": np.array(src.info["topology"]),
        "curves": np.array(curves),
        "source_archive_version": np.int64(src.info["archive_version"]),
        "combinations": np.array([_combo_tag(*c) for c in combos]),
        "resolution_used": src.resolution,
        "n_points_mean_used": src.n_points_mean,
        "length_mean_used": src.length_mean,
        "length_std_used": src.length_std,
        "pivot_internal_np": np.float64(np.exp(ln_xc_int) + 1.0),
    }
    if curves == "realisation":
        archive["source_realisation_indices"] = run["source_indices"]
        archive["n_source_realisations"] = np.int64(run["n_target"])
        archive["n_fitted_realisations"] = np.int64(run["n_valid"])
    for scheme in weight_schemes:
        if scheme != "none":
            for fit in fits:
                archive[f"weights_{_combo_tag(fit, scheme)}"] = weights[(fit, scheme)]

    summary_rows: List[List] = []
    diagnostic_rows: List[List] = []
    combo_meta: Dict[str, Dict] = {}
    for c in combos:
        tag = _combo_tag(*c)
        s = store[c]
        beta = s["beta"]
        Df, L0, A = _derived(beta, s["ln_a"], ln_xc_int, ln_xc_rep[c])
        samples = {"beta": beta, "Df": Df, "L0": L0, "A": A, "objective_per_dof": s["obj"] / dof}

        mc = mean_curve[c]
        mDf, mL0, mA = _derived(mc["beta"], mc["ln_a"], ln_xc_int, ln_xc_rep[c])
        point = {
            "beta": float(mc["beta"][0]), "Df": float(mDf[0]), "L0": float(mL0[0]),
            "A": float(mA[0]), "objective_per_dof": float(mc["obj"][0] / dof),
        }

        pivot_np = float(np.exp(ln_xc_rep[c]) + 1.0)
        archive[f"{tag}_pivot_np"] = np.float64(pivot_np)
        stats = {}
        for name in _PARAMETERS:
            archive[f"{tag}_{name}_samples"] = samples[name]
            archive[f"{tag}_{name}_mean_curve"] = np.float64(point[name])
            stats[name] = _summary(samples[name])
            if name == "objective_per_dof":
                continue
            h = _pdf1d(samples[name], output_bins, hist_tail)
            if h is not None:
                archive[f"{tag}_{name}_hist_counts"] = h[0]
                archive[f"{tag}_{name}_hist_density"] = h[1]
                archive[f"{tag}_{name}_bin_edges"] = h[2]
                archive[f"{tag}_{name}_hist_underflow"] = np.int64(h[3])
                archive[f"{tag}_{name}_hist_overflow"] = np.int64(h[4])
        for name in _PARAMETERS:
            st = stats[name]
            summary_rows.append([
                c[0], c[1], name, st["n_finite"], st["mean"], st["std"], st["median"],
                st["p2_5"], st["p16"], st["p84"], st["p97_5"], point[name],
            ])

        for amp_name in ("L0", "A"):
            j = _pdf2d(samples[amp_name], Df, joint_bins, hist_tail)
            if j is not None:
                archive[f"{tag}_joint_{amp_name}_Df_hist_counts"] = j[0]
                archive[f"{tag}_joint_{amp_name}_Df_hist_density"] = j[1]
                archive[f"{tag}_joint_{amp_name}_Df_{amp_name}_bin_edges"] = j[2]
                archive[f"{tag}_joint_{amp_name}_Df_Df_bin_edges"] = j[3]
                archive[f"{tag}_joint_{amp_name}_Df_n_outside"] = np.int64(j[4])

        corr = {
            "corr_L0_beta": _corr(L0, beta),
            "corr_A_beta": _corr(A, beta),
            "corr_L0_Df": _corr(L0, Df),
            "corr_A_Df": _corr(A, Df),
            "spearman_L0_beta": _spearman(L0, beta),
            "spearman_A_beta": _spearman(A, beta),
        }
        n_eff = _n_eff(weights[c])
        frac_neg = float(np.mean(beta < 0.0))
        frac_ge1 = float(np.mean(beta >= 1.0))
        n_at_bound = int(s["at_bound"].sum())
        for k, v in corr.items():
            archive[f"{tag}_{k}"] = np.float64(v)
        archive[f"{tag}_n_eff_weights"] = np.float64(n_eff)
        archive[f"{tag}_frac_beta_negative"] = np.float64(frac_neg)
        archive[f"{tag}_frac_beta_ge_1"] = np.float64(frac_ge1)
        archive[f"{tag}_n_at_beta_bound"] = np.int64(n_at_bound)

        val = validation.get(c)
        has_val = val is not None and val["n"] > 0
        if has_val:
            archive[f"{tag}_validation_n"] = np.int64(val["n"])
            archive[f"{tag}_validation_n_fail"] = np.int64(val["n_fail"])
            archive[f"{tag}_validation_max_abs_dbeta"] = np.float64(val["max_abs_dbeta"])
            archive[f"{tag}_validation_max_rel_dA"] = np.float64(val["max_rel_dA"])
        val_cols = (
            [val["n"], val["n_fail"], val["max_abs_dbeta"], val["max_rel_dA"]]
            if has_val else [0, 0, float("nan"), float("nan")]
        )
        diagnostic_rows.append([
            c[0], c[1], pivot_np, corr["corr_L0_beta"], corr["corr_A_beta"],
            corr["corr_L0_Df"], corr["corr_A_Df"], frac_neg, frac_ge1, n_at_bound,
            *val_cols, corr["spearman_L0_beta"], corr["spearman_A_beta"], n_eff,
        ])
        combo_meta[tag] = {
            "fit": c[0], "weights": c[1], "pivot_np": pivot_np,
            "summary": stats, "mean_curve_fit": point, **corr,
            "n_eff_weights": n_eff,
            "frac_beta_negative": frac_neg, "frac_beta_ge_1": frac_ge1,
            "n_at_beta_bound": n_at_bound,
            "validation": val if has_val else None,
        }

    try:
        import scipy
        scipy_version = scipy.__version__
    except ImportError:  # pragma: no cover
        scipy_version = None

    metadata = {
        "mc_archive_version": MC_ARCHIVE_VERSION,
        "source_npz": str(src.path),
        "source": {k: v for k, v in src.info.items() if k != "metadata"},
        "source_metadata": src.info["metadata"],
        "method": src.method,
        "topology": src.info["topology"],
        "interpretation": (
            "P(L|n_p) is the distribution of the lengths the method can return (its spread is "
            "the variability of the method, not a measurement error); the fitted-parameter "
            "distributions are the values the method can produce"
        ),
        "model": "L(n_p) = L0 * (n_p - 1)^beta = A * ((n_p - 1)/x_c)^beta, beta = 1 - 1/D",
        "fit_equations": {
            "loglog": "ln L = ln A + beta ln((n_p - 1)/x_c)",
            "nonlinear": "L = A ((n_p - 1)/x_c)^beta, least squares in L",
        },
        "curves": curves,
        "curves_note": _CURVES_NOTES[(src.method, curves)],
        "weights_note": (
            "inverse-variance: nonlinear w=1/var(L), loglog w=mean^2/var(L) (delta method); "
            "var(L) is the spread of the method at each resolution, fixed per resolution; "
            "YS uses var(L|d) only"
        ),
        "pivot_note": {
            "auto": "empirical: cov(ln A, beta) = 0 over the accepted fits of each combination",
            "weights": "geometric mean of x with log-space-equivalent weights",
        }.get(pivot, "fixed by the user"),
        "beta_bounds_note": "beta_min/beta_max apply to nonlinear fits only",
        "objective_note": (
            "objective_per_dof = weighted residual sum / (n_res - 2); a goodness-of-fit summary "
            "of each curve, not a chi^2 (the spread of P(L|n_p) is not a measurement error)"
        ),
        "histogram_note": (
            f"marginal and joint histograms span the [{hist_tail:g}, {1 - hist_tail:g}] "
            "quantiles; samples outside are counted in *_underflow/_overflow/_n_outside; "
            "density is normalised by the number of finite samples"
        ),
        "notes": notes,
        "combinations": [_combo_tag(*c) for c in combos],
        "fits": fits,
        "weights": weight_schemes,
        "pivot": pivot,
        "pivot_internal_np": float(np.exp(ln_xc_int) + 1.0),
        "nonlinear_solver": nonlinear_solver,
        "validate_curve_fit": validate_curve_fit,
        "curve_fit_tol": _CURVE_FIT_TOL,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "beta_grid": beta_grid,
        "beta_xtol": beta_xtol,
        "hist_tail": hist_tail,
        "n_curves": run["n_valid"],
        "n_sim_requested": n_sim if curves != "realisation" else None,
        "n_source_realisations": run["n_target"] if curves == "realisation" else None,
        "n_attempted": run["n_attempted"],
        "n_rejected_invalid_data": run["n_rejected_data"],
        "n_rejected_fit_failure": run["n_rejected_fit"],
        "seed": seed if curves != "realisation" else None,
        "np_min": np_min,
        "np_max": np_max,
        "n_resolutions_used": int(n_res),
        "resolution_values": src.resolution.tolist(),
        "n_points_mean": src.n_points_mean.tolist(),
        "results": combo_meta,
        "wall_seconds": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy_version,
            "platform": platform.platform(),
        },
    }
    archive["metadata_json"] = np.array(json.dumps(metadata))

    paths = _write(Path(output_dir), prefix, archive, summary_rows, diagnostic_rows)
    if verbose:
        _report(src, curves, combos, combo_meta, run, n_sim, nonlinear_solver,
                ln_xc_int, pivot, paths, metadata["wall_seconds"])
    return archive


def _write(out: Path, prefix: str, archive, summary_rows, diagnostic_rows) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / f"{prefix}_mc_fit.npz"
    np.savez_compressed(npz_path, **archive)

    summary_path = out / f"{prefix}_mc_summary.csv"
    with summary_path.open("w") as f:
        f.write(
            "fit,weights,parameter,n_finite,mean,std,median,p2_5,p16,p84,p97_5,mean_curve_fit\n"
        )
        for r in summary_rows:
            f.write(",".join([r[0], r[1], r[2], str(r[3])] + [f"{v:.12g}" for v in r[4:]]) + "\n")

    diag_path = out / f"{prefix}_mc_diagnostics.csv"
    with diag_path.open("w") as f:
        f.write(
            "fit,weights,pivot_np,corr_L0_beta,corr_A_beta,corr_L0_Df,corr_A_Df,"
            "frac_beta_negative,frac_beta_ge_1,n_at_beta_bound,"
            "validation_n,validation_n_fail,validation_max_abs_dbeta,validation_max_rel_dA,"
            "spearman_L0_beta,spearman_A_beta,n_eff_weights\n"
        )
        for r in diagnostic_rows:
            f.write(
                ",".join(
                    [r[0], r[1]]
                    + [f"{v:.12g}" for v in r[2:9]]
                    + [str(r[9]), str(r[10]), str(r[11])]
                    + [f"{v:.12g}" for v in r[12:]]
                )
                + "\n"
            )
    return [npz_path, summary_path, diag_path]


def _report(src, curves, combos, combo_meta, run, n_sim, nonlinear_solver, ln_xc_int,
            pivot, paths, wall) -> None:
    count_text = (
        f"N_sim={n_sim}" if curves != "realisation"
        else f"stored realisations={run['n_target']}, fitted={run['n_valid']}"
    )
    print(
        f"method: {src.method.upper()} ({curves} curves) | resolutions used: "
        f"{src.resolution.size} | {count_text} (attempted {run['n_attempted']}, rejected "
        f"data={run['n_rejected_data']}, fit={run['n_rejected_fit']})"
    )
    print(
        f"nonlinear solver: {nonlinear_solver} | internal pivot n_p = "
        f"{np.exp(ln_xc_int) + 1:.6g} | reported pivot: {pivot}"
    )
    for c in combos:
        cm = combo_meta[_combo_tag(*c)]
        sb, sd, sa = cm["summary"]["beta"], cm["summary"]["Df"], cm["summary"]["A"]
        print(f"\n[{_combo_tag(*c)}]  pivot n_p = {cm['pivot_np']:.6g} | "
              f"effective resolutions = {cm['n_eff_weights']:.1f} of {src.resolution.size}")
        print(f"  beta = {sb['median']:.6g} [{sb['p16']:.6g}, {sb['p84']:.6g}] (16-84%)")
        print(f"  Df   = {sd['median']:.6g} [{sd['p16']:.6g}, {sd['p84']:.6g}] (16-84%)")
        print(f"  A    = {sa['median']:.6g} [{sa['p16']:.6g}, {sa['p84']:.6g}] (16-84%)")
        print(
            f"  corr(L0,beta) = {cm['corr_L0_beta']:.4f} | corr(A,beta) = {cm['corr_A_beta']:.4f}"
            f" | spearman(A,beta) = {cm['spearman_A_beta']:.4f}"
        )
        print(
            f"  frac beta<0 = {cm['frac_beta_negative']:.4g} | frac beta>=1 = "
            f"{cm['frac_beta_ge_1']:.4g} | at bound = {cm['n_at_beta_bound']}"
        )
        if cm["validation"] is not None:
            v = cm["validation"]
            print(
                f"  validation vs {run['other_solver']} on {v['n']} curves: "
                f"max|dbeta| = {v['max_abs_dbeta']:.3g}, max|dA/A| = {v['max_rel_dA']:.3g}, "
                f"failures = {v['n_fail']}"
            )
    print(f"\nwrote {paths[0]}")
    for p in paths[1:]:
        print(f"      {p}")
    print(f"elapsed {wall:.1f} s")


# --------------------------------------------------------------------------- CLI
def _pivot_arg(value: str) -> str:
    if value in ("auto", "weights"):
        return value
    try:
        v = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("pivot must be 'auto', 'weights' or a number > 1")
    if not v > 1.0:
        raise argparse.ArgumentTypeError("pivot must be 'auto', 'weights' or a number > 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coastline_mc_fit",
        description="Monte-Carlo fit of the power law L(n_p) = L0 (n_p - 1)^beta, "
                    "beta = 1 - 1/D, from a coastline_run.py archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="<prefix>_pdfs.npz written by coastline_run.py (archive version >= 6)")
    p.add_argument(
        "--curves", required=True, choices=CURVES_CHOICES,
        help="how curves are built: 'histogram' (RP only) and 'samples' draw each "
             "resolution independently, from the saved histograms or from the stored "
             "samples; 'realisation' fits every stored realisation once, keeping the "
             "correlation between resolutions (RP: --rp-sweep nested --save-samples; "
             "YS: --ys-seeds common)",
    )
    p.add_argument("--n-sim", type=int, default=100_000,
                   help="accepted curves for 'histogram' and 'samples'; ignored by 'realisation'")
    p.add_argument("--seed", type=int, default=20260814,
                   help="RNG seed for 'histogram' and 'samples'; ignored by 'realisation'")
    p.add_argument("--np-min", type=float,
                   help="smallest n_p kept in the fit (YS: mean n_p of each ruler)")
    p.add_argument("--np-max", type=float,
                   help="largest n_p kept in the fit (YS: mean n_p of each ruler)")

    g = p.add_argument_group("fit")
    g.add_argument("--fit", nargs="+", choices=FIT_CHOICES, default=["loglog"],
                   help="estimator(s); all run on the same curves")
    g.add_argument("--weights", nargs="+", choices=WEIGHT_CHOICES, default=["none"],
                   help="weighting scheme(s), applied to every --fit")
    g.add_argument(
        "--pivot", type=_pivot_arg, default="auto",
        help="n_p at which A is reported: 'auto' makes cov(ln A, beta) = 0; 'weights' = "
             "weighted geometric mean of n_p - 1; a number fixes it (2 gives A = L0)",
    )
    g.add_argument(
        "--nonlinear-solver", choices=SOLVER_CHOICES, default="profile",
        help="'profile' = Newton on the profiled objective, grid fallback; 'profile-grid' "
             "= grid + bisection; 'curve_fit' = scipy, one call per curve",
    )
    g.add_argument(
        "--validate-curve-fit", type=int, default=0,
        help="re-fit the first N accepted curves with curve_fit (with profile when "
             "curve_fit is the solver) and report the differences",
    )
    g.add_argument("--beta-min", type=float, help="lower bound on beta (nonlinear only)")
    g.add_argument("--beta-max", type=float, help="upper bound on beta (nonlinear only)")
    g.add_argument("--beta-grid", type=int, default=32,
                   help="points in the coarse beta grid of the grid method")
    g.add_argument("--beta-xtol", type=float, default=1e-10, help="convergence tolerance in beta")

    g = p.add_argument_group("output")
    g.add_argument("--output-bins", type=int, default=100, help="bins of the marginal PDFs")
    g.add_argument("--joint-bins", type=int, default=80, help="bins per axis of the joint PDFs")
    g.add_argument(
        "--hist-tail", type=float, default=1e-3,
        help="histograms span the [tail, 1 - tail] quantiles (0 = full range); samples "
             "outside are counted",
    )
    g.add_argument("--output-dir", type=Path, default=Path("results/mc_fit"))
    g.add_argument("--prefix", type=str,
                   help="file prefix [default: <input prefix>_<curves>, e.g. run_samples]")

    g = p.add_argument_group("execution")
    g.add_argument("--chunk-elements", type=int, default=1_048_576,
                   help="memory knob: about this many (curve x resolution) values per "
                        "chunk; results do not depend on it")
    g.add_argument("--chunk-sim", type=int,
                   help="curves per chunk (overrides --chunk-elements; results do not "
                        "depend on it)")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_mc(
        input_npz=args.input,
        curves=args.curves,
        n_sim=args.n_sim,
        seed=args.seed,
        np_min=args.np_min,
        np_max=args.np_max,
        output_bins=args.output_bins,
        joint_bins=args.joint_bins,
        output_dir=args.output_dir,
        prefix=args.prefix,
        fits=args.fit,
        weight_schemes=args.weights,
        pivot=args.pivot,
        nonlinear_solver=args.nonlinear_solver,
        validate_curve_fit=args.validate_curve_fit,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_grid=args.beta_grid,
        beta_xtol=args.beta_xtol,
        chunk_elements=args.chunk_elements,
        chunk_sim=args.chunk_sim,
        hist_tail=args.hist_tail,
    )


if __name__ == "__main__":
    main()
