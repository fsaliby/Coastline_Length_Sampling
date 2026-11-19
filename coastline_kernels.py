"""Compiled (numba) kernels for the estimators. Optional: imported only for ``engine="numba"``.

Every kernel evaluates the *same* curve as :class:`~coastline_geometry.CoastlineModel`:
it reads the piecewise-polynomial coefficients of the model's interpolants and
evaluates them with the same Horner recurrence SciPy uses, so a station placed at
arc length ``s`` lands on the same point of the same curve. What changes is how the
work is organised: one fused loop per realisation (``s -> t -> (lat, lon) -> chord``)
instead of a chain of whole-array NumPy operations, hence no temporaries and no
per-call overhead.

Interval search. Two strategies, chosen by measurement:

* Random points: stations are spread over the whole curve (hundreds to thousands of
  map cells between neighbours for n_points <~ 1000), so a hint is of little use. The
  cell of ``s`` is read from a uniform bucket index (``bidx``: map cell at the left
  edge of every bucket of width ``1 / inv_h``) and fixed up with a bounded bisection,
  O(1) on average. Each map cell also stores the knot interval it lies in (cells
  never straddle a knot), which removes the second search altogether.
* Yardstick: successive trials of one step land close to the previous one, so every
  search starts from an interval hint and gallops outwards (steps 1, 2, 4, ...)
  before bisecting. The bucket index was measured and brings nothing there.

Every search returns the same interval as a plain bisection, so results do not depend
on the strategy.

Memory layout. The random-points kernels read *row-wise* tables (one row per map
cell / knot interval holding the breakpoint and all the coefficients), so that a
station touches one or two cache lines per table instead of one per coefficient, as
it would with SciPy's ``c[order, interval]`` layout. The yardstick kernel keeps
SciPy's layout: its accesses are local and already cached.

Polyline walker (``ys_walk_polyline``). For the piecewise-linear curve only. It does
not use the interpolants at all: it reads the vertices and their cumulative chordal
distance, and locates every station from the convexity of the divider circle (see
:mod:`coastline_estimators`). Only vertices are tested, so there is no search to
organise; the path-length skip is a short forward (or backward) scan of ``tk``.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import numba as nb
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(
        "engine='numba' needs the numba package (pip install numba)"
    ) from exc

from scipy.interpolate import BSpline, PPoly

__all__ = ["KernelTables", "rp_independent", "rp_independent_closed", "rp_nested", "rp_nested_closed",
           "ys_walk", "ys_walk_polyline", "ys_walk_polyline_closed"]


# --------------------------------------------------------------------- tables
def _as_ppoly(f) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Breakpoints and coefficients of an interpolant, without zero-length pieces."""
    if isinstance(f, BSpline):
        f = PPoly.from_spline(f)
    x = np.asarray(f.x, dtype=float)
    c = np.asarray(f.c, dtype=float)
    if c.ndim != 2:
        raise ValueError("only scalar-valued interpolants are supported")
    keep = np.diff(x) > 0.0
    x = np.concatenate([x[:-1][keep], x[-1:]])
    c = np.ascontiguousarray(c[:, keep])
    return x, c


_BUCKETS_PER_CELL = 2       # bucket-index resolution (measured optimum: 1-4)


class KernelTables:
    """Plain arrays describing the model, in the form the kernels consume."""

    def __init__(self, model) -> None:
        self.xs, self.cs = _as_ppoly(model._t_of_s)
        xk_lat, self.clat = _as_ppoly(model._lat_spl)
        xk_lon, self.clon = _as_ppoly(model._lon_spl)
        if not np.array_equal(xk_lat, xk_lon):
            raise ValueError("lat and lon interpolants must share their breakpoints")
        self.xk = xk_lat
        self.radius = float(model.radius)
        self.length = float(model.length)
        self._build_rowwise()

    def _build_rowwise(self) -> None:
        """Row-wise tables and bucket index for the random-points kernels.

        ``smap[i]``  = [s_i, Horner coefficients of t(s) on cell i, knot interval of cell i]
        ``curve[j]`` = [t_j, Horner coefficients of lat, Horner coefficients of lon]
        ``bidx[k]``  = map cell containing s = k / inv_h
        The last row of ``smap`` / ``curve`` only carries the final breakpoint.
        """
        xs, cs, xk = self.xs, self.cs, self.xk
        n_cells, n_iv = xs.size - 1, xk.size - 1
        ks, kk = cs.shape[0], self.clat.shape[0]
        if self.clon.shape[0] != kk:
            raise ValueError("lat and lon interpolants must have the same order")

        # A map cell lies inside one knot interval; its midpoint identifies it safely.
        t_mid = 0.5 * (cs[-1, :] + np.append(cs[-1, 1:], xk[-1]))
        cell_iv = np.clip(np.searchsorted(xk, t_mid, side="right") - 1, 0, n_iv - 1)

        smap = np.zeros((n_cells + 1, ks + 2))
        smap[:, 0] = xs
        smap[:n_cells, 1:ks + 1] = cs.T
        smap[:n_cells, ks + 1] = cell_iv
        smap[n_cells, ks + 1] = cell_iv[-1]

        curve = np.zeros((n_iv + 1, 1 + 2 * kk))
        curve[:, 0] = xk
        curve[:n_iv, 1:1 + kk] = self.clat.T
        curve[:n_iv, 1 + kk:] = self.clon.T

        n_buckets = _BUCKETS_PER_CELL * n_cells
        self.inv_h = n_buckets / (xs[-1] - xs[0])
        edges = xs[0] + np.arange(n_buckets + 1) / self.inv_h
        self.bidx = np.clip(
            np.searchsorted(xs, edges, side="right") - 1, 0, n_cells - 1
        ).astype(np.int32)
        self.smap, self.curve = smap, curve

    def map_args(self):
        """Curve and map arrays of the yardstick kernel, in argument order."""
        return (self.xs, self.cs, self.xk, self.clat, self.clon)

    def rp_args(self, use_map: bool):
        """Arguments of the random-points kernels, in argument order."""
        return (self.smap, self.curve, self.bidx, self.inv_h, self.xk, use_map)


# ------------------------------------------------------------------ primitives
@nb.njit(cache=True, inline="always")
def _horner(c, i, d):
    r = c[0, i]
    for q in range(1, c.shape[0]):
        r = r * d + c[q, i]
    return r


@nb.njit(cache=True, inline="always")
def _advance(xb, i, v):
    """Interval of ``v`` in ``xb`` for non-decreasing queries (``v >= xb[i]``):
    gallop forward from the hint ``i``, then bisect."""
    n = xb.size
    if i >= n - 2 or v < xb[i + 1]:
        return i
    lo = i + 1
    step = 1
    hi = lo + 1
    while hi < n - 1 and v >= xb[hi]:
        lo = hi
        step <<= 1
        hi = lo + step
    if hi > n - 1:
        hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if v >= xb[mid]:
            lo = mid
        else:
            hi = mid
    return min(lo, n - 2)


@nb.njit(cache=True, inline="always")
def _seek(xb, i, v):
    """Interval of ``v`` in ``xb`` starting from hint ``i``, in either direction."""
    n = xb.size
    if i > n - 2:
        i = n - 2
    if v >= xb[i]:
        return _advance(xb, i, v)
    hi = i
    step = 1
    lo = hi - 1
    while lo > 0 and v < xb[lo]:
        hi = lo
        step <<= 1
        lo = hi - step
    if lo < 0:
        lo = 0
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if v >= xb[mid]:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit(cache=True, inline="always")
def _locate(xb, v):
    """Interval of ``v`` in ``xb`` for arbitrary queries (bisection)."""
    lo, hi = 0, xb.size - 1
    while hi - lo > 1:
        mid = (lo + hi) >> 1
        if v >= xb[mid]:
            lo = mid
        else:
            hi = mid
    return lo


@nb.njit(cache=True, inline="always")
def _cell(smap, bidx, inv_h, s):
    """Map cell of arc length ``s`` (queries in any order): bucket index, then a
    bisection bounded by the cells of that bucket. The two closing loops absorb the
    round-off of the bucket computation at bucket edges."""
    n = smap.shape[0] - 1
    k = int((s - smap[0, 0]) * inv_h)
    if k < 0:
        k = 0
    elif k > bidx.size - 2:
        k = bidx.size - 2
    lo = bidx[k]
    hi = bidx[k + 1]
    while hi > lo:
        mid = (lo + hi + 1) >> 1
        if s >= smap[mid, 0]:
            lo = mid
        else:
            hi = mid - 1
    while lo > 0 and s < smap[lo, 0]:
        lo -= 1
    while lo < n - 1 and s >= smap[lo + 1, 0]:
        lo += 1
    return lo


@nb.njit(cache=True, inline="always")
def _curve_at(curve, j, t):
    """(lat, lon) at chordal parameter ``t`` lying in knot interval ``j``."""
    kk = (curve.shape[1] - 1) >> 1
    dt = t - curve[j, 0]
    la = curve[j, 1]
    lo = curve[j, 1 + kk]
    for q in range(1, kk):
        la = la * dt + curve[j, 1 + q]
        lo = lo * dt + curve[j, 1 + kk + q]
    return la, lo


@nb.njit(cache=True, inline="always")
def _latlon_s(s, smap, curve, bidx, inv_h):
    """(lat, lon) at arc length ``s``; queries in any order."""
    ks = smap.shape[1] - 2
    i = _cell(smap, bidx, inv_h, s)
    ds = s - smap[i, 0]
    t = smap[i, 1]
    for q in range(2, ks + 1):
        t = t * ds + smap[i, q]
    last = curve.shape[0] - 1
    t = min(max(t, curve[0, 0]), curve[last, 0])
    # Knot interval of the cell; the last ulp of t(s) may step over a knot.
    j = int(smap[i, ks + 1])
    while j > 0 and t < curve[j, 0]:
        j -= 1
    while j < last - 1 and t >= curve[j + 1, 0]:
        j += 1
    la, lo = _curve_at(curve, j, t)
    return la, lo


@nb.njit(cache=True, inline="always")
def _latlon_t(t, j, xk, curve):
    """(lat, lon, j) at chordal parameter ``t``. ``j`` is an interval hint that only
    moves forward (queries must be non-decreasing, as along a sorted row)."""
    j = _advance(xk, j, t)
    la, lo = _curve_at(curve, j, t)
    return la, lo, j


@nb.njit(cache=True, inline="always")
def _latlon_near(s, i, j, xs, cs, xk, clat, clon):
    """(lat, lon, i, j) at arc length ``s``, queries in any order; ``i``, ``j`` are hints."""
    i = _seek(xs, i, s)
    t = _horner(cs, i, s - xs[i])
    t = min(max(t, xk[0]), xk[-1])
    j = _seek(xk, j, t)
    dt = t - xk[j]
    return _horner(clat, j, dt), _horner(clon, j, dt), i, j


@nb.njit(cache=True, inline="always")
def _haversine(lat1, lon1, lat2, lon2, R):
    a = math.sin(0.5 * (lat2 - lat1))
    b = math.sin(0.5 * (lon2 - lon1))
    h = a * a + math.cos(lat1) * math.cos(lat2) * b * b
    return 2.0 * R * math.asin(math.sqrt(min(h, 1.0)))


@nb.njit(cache=True, inline="always")
def _haversine_c(lat1, lon1, cos1, lat2, lon2, R):
    """:func:`_haversine` with ``cos(lat1)`` supplied (the fixed end of many chords)."""
    a = math.sin(0.5 * (lat2 - lat1))
    b = math.sin(0.5 * (lon2 - lon1))
    h = a * a + cos1 * math.cos(lat2) * b * b
    return 2.0 * R * math.asin(math.sqrt(min(h, 1.0)))


@nb.njit(cache=True, inline="always")
def _chord3(V, a, b, R):
    dx = V[a, 0] - V[b, 0]
    dy = V[a, 1] - V[b, 1]
    dz = V[a, 2] - V[b, 2]
    return 2.0 * R * math.asin(min(0.5 * math.sqrt(dx * dx + dy * dy + dz * dz), 1.0))


# ---------------------------------------------------------------- random points
@nb.njit(cache=True)
def rp_independent(P, smap, curve, bidx, inv_h, xk, use_map, R):
    """Polyline length of every row of ``P`` (stations sorted along each row)."""
    m, n = P.shape
    out = np.empty(m)
    for r in range(m):
        j = 0
        acc = 0.0
        px = py = pz = 0.0
        for c in range(n):
            if use_map:
                la, lo = _latlon_s(P[r, c], smap, curve, bidx, inv_h)
            else:
                la, lo, j = _latlon_t(P[r, c], j, xk, curve)
            cl = math.cos(la)
            x = cl * math.cos(lo)
            y = cl * math.sin(lo)
            z = math.sin(la)
            if c > 0:
                dx = x - px
                dy = y - py
                dz = z - pz
                acc += 2.0 * R * math.asin(min(0.5 * math.sqrt(dx * dx + dy * dy + dz * dz), 1.0))
            px, py, pz = x, y, z
        out[r] = acc
    return out


@nb.njit(cache=True)
def rp_independent_closed(P, smap, curve, bidx, inv_h, xk, use_map, R):
    """Closed polygon, n unique stations connected cyclically (no fixed vertex)."""
    m, n = P.shape
    out = np.empty(m)
    for r in range(m):
        j = 0
        acc = 0.0
        x0 = y0 = z0 = 0.0
        px = py = pz = 0.0
        for c in range(n):
            if use_map:
                la, lo = _latlon_s(P[r, c], smap, curve, bidx, inv_h)
            else:
                la, lo, j = _latlon_t(P[r, c], j, xk, curve)
            cl = math.cos(la)
            x, y, z = cl * math.cos(lo), cl * math.sin(lo), math.sin(la)
            if c == 0:
                x0, y0, z0 = x, y, z
            else:
                dx, dy, dz = x - px, y - py, z - pz
                acc += 2.0 * R * math.asin(min(0.5 * math.sqrt(dx*dx + dy*dy + dz*dz), 1.0))
            px, py, pz = x, y, z
        dx, dy, dz = x0 - px, y0 - py, z0 - pz
        out[r] = acc + 2.0 * R * math.asin(min(0.5 * math.sqrt(dx*dx + dy*dy + dz*dz), 1.0))
    return out


@nb.njit(cache=True)
def rp_nested_closed(U, smap, curve, bidx, inv_h, xk, use_map, R, cols, out):
    """Reverse deletion from a circular doubly linked list; one draw per row."""
    m, N = U.shape
    ps = np.empty(N)
    V = np.empty((N, 3))
    prv = np.empty(N, np.int64)
    nxt = np.empty(N, np.int64)
    rc = np.empty(N)
    rank = np.empty(N, np.int64)
    for r in range(m):
        order = np.argsort(U[r])
        for q in range(N):
            rank[order[q]] = q
            ps[q] = U[r, order[q]]
        j = 0
        for c in range(N):
            if use_map:
                la, lo = _latlon_s(ps[c], smap, curve, bidx, inv_h)
            else:
                la, lo, j = _latlon_t(ps[c], j, xk, curve)
            cl = math.cos(la)
            V[c, 0] = cl * math.cos(lo)
            V[c, 1] = cl * math.sin(lo)
            V[c, 2] = math.sin(la)
        tot = 0.0
        for c in range(N):
            b = (c + 1) % N
            rc[c] = _chord3(V, c, b, R)
            tot += rc[c]
            nxt[c] = b
            prv[b] = c
        if cols[N + 1] >= 0:
            out[r, cols[N + 1]] = tot
        for k in range(N - 1, 1, -1):
            x = rank[k]
            a, b = prv[x], nxt[x]
            new = _chord3(V, a, b, R)
            tot += new - rc[a] - rc[x]
            rc[a] = new
            nxt[a] = b
            prv[b] = a
            if cols[k + 1] >= 0:
                out[r, cols[k + 1]] = tot


@nb.njit(cache=True)
def rp_nested(U, p_lo, p_hi, smap, curve, bidx, inv_h, xk, use_map, R, cols, out):
    """Nested random-points sweep.

    ``U[r]`` holds the interior stations of realisation ``r`` in *insertion order*.
    The configuration with ``n`` points is the two endpoints plus ``U[r, :n-2]``,
    so every ``n`` is an exact draw of ``n-2`` uniform interior points. All stations
    are placed once; the points are then removed from last to first through a
    linked list, each removal costing one new chord. ``cols[n]`` is the output column
    for ``n`` points, or -1 when ``n`` is not requested. ``out[r, -1]`` receives the
    running total at n = 2, which the caller compares with the direct chord as a
    round-off check.
    """
    m, K = U.shape
    N = K + 2
    ps = np.empty(N)
    V = np.empty((N, 3))
    prv = np.empty(N, np.int64)
    nxt = np.empty(N, np.int64)
    rc = np.empty(N)  # chord from each node to its current successor
    rank = np.empty(K, np.int64)
    for r in range(m):
        order = np.argsort(U[r])
        for q in range(K):
            rank[order[q]] = q + 1
            ps[q + 1] = U[r, order[q]]
        ps[0] = p_lo
        ps[N - 1] = p_hi
        j = 0
        for c in range(N):
            if use_map:
                la, lo = _latlon_s(ps[c], smap, curve, bidx, inv_h)
            else:
                la, lo, j = _latlon_t(ps[c], j, xk, curve)
            cl = math.cos(la)
            V[c, 0] = cl * math.cos(lo)
            V[c, 1] = cl * math.sin(lo)
            V[c, 2] = math.sin(la)
        tot = 0.0
        for c in range(N - 1):
            rc[c] = _chord3(V, c, c + 1, R)
            tot += rc[c]
            nxt[c] = c + 1
            prv[c + 1] = c
        if cols[N] >= 0:
            out[r, cols[N]] = tot
        for k in range(K - 1, -1, -1):
            x = rank[k]
            a = prv[x]
            b = nxt[x]
            new = _chord3(V, a, b, R)
            tot += new - rc[a] - rc[x]
            rc[a] = new
            nxt[a] = b
            prv[b] = a
            if cols[k + 2] >= 0:
                out[r, cols[k + 2]] = tot
        out[r, out.shape[1] - 1] = tot


# -------------------------------------------------------------------- yardstick
@nb.njit(cache=True)
def _ys_one(s0, direction, d, lip, ftol, xtol, hmin, max_iter, L, R, xs, cs, xk, clat, clon):
    """One walker. Returns (steps, chord_sum, last_chord, s_end, s_prev).

    ``hi_s`` and ``hi_k`` follow the walker as interval hints for every evaluation.
    """
    s_base = s0
    hi_s = _locate(xs, s_base)
    hi_k = 0
    lat_b, lon_b, hi_s, hi_k = _latlon_near(s_base, hi_s, hi_k, xs, cs, xk, clat, clon)
    steps = 0
    chord_sum = 0.0
    last_chord = 0.0
    s_prev = s_base
    while True:
        dmax = (L - s_base) if direction > 0 else s_base
        if lip * dmax < d:          # no chord of length d fits in what is left
            break
        # --- march: Lipschitz-safe steps, never past the first crossing
        lo = 0.0
        flo = -d
        trial = min(d / lip, dmax)
        found = False
        crossed = False
        f = 0.0
        la = lb = 0.0
        while True:
            la, lb, hi_s, hi_k = _latlon_near(
                s_base + direction * trial, hi_s, hi_k, xs, cs, xk, clat, clon
            )
            f = _haversine(lat_b, lon_b, la, lb, R) - d
            if abs(f) <= ftol:
                found = True
                break
            if f > 0.0:
                crossed = True
                break
            if trial >= dmax:
                break
            lo = trial
            flo = f
            trial = min(trial + max(-f / lip, hmin), dmax)
        if not (found or crossed):
            break
        # --- bracket: Illinois on [lo, trial]
        if crossed:
            a = lo
            fa = flo
            b = trial
            fb = f
            side = 0
            for _ in range(max_iter):
                c = b - fb * (b - a) / (fb - fa)
                if not (a < c < b):
                    c = 0.5 * (a + b)
                la, lb, hi_s, hi_k = _latlon_near(
                    s_base + direction * c, hi_s, hi_k, xs, cs, xk, clat, clon
                )
                f = _haversine(lat_b, lon_b, la, lb, R) - d
                trial = c
                if abs(f) <= ftol:
                    break
                if f > 0.0:
                    b = c
                    fb = f
                    if side == 1:
                        fa *= 0.5
                    side = 1
                else:
                    a = c
                    fa = f
                    if side == -1:
                        fb *= 0.5
                    side = -1
                if b - a <= xtol:
                    break
        chord = d + f
        steps += 1
        chord_sum += chord
        last_chord = chord
        s_prev = s_base
        s_base = s_base + direction * trial
        lat_b = la
        lon_b = lb
    return steps, chord_sum, last_chord, s_base, s_prev


@nb.njit(cache=True)
def ys_walk(s0, direction, d, lip, ftol, xtol, hmin, max_iter, L, R, xs, cs, xk, clat, clon):
    n = s0.size
    steps = np.empty(n, np.int64)
    csum = np.empty(n)
    last = np.empty(n)
    s_end = np.empty(n)
    s_prev = np.empty(n)
    for w in range(n):
        st, cs_, la, se, sp = _ys_one(
            s0[w], direction, d, lip, ftol, xtol, hmin, max_iter, L, R, xs, cs, xk, clat, clon
        )
        steps[w] = st
        csum[w] = cs_
        last[w] = la
        s_end[w] = se
        s_prev[w] = sp
    return steps, csum, last, s_end, s_prev


# ----------------------------------------------------------- yardstick, polyline
@nb.njit(cache=True)
def _ys_polyline_one(plat, plon, seg, direction, d, ftol, xtol, max_iter, R, vlat, vlon, tk):
    """One walker on the polyline with vertices ``(vlat, vlon)`` and cumulative chordal
    distance ``tk``, started at ``(plat, plon)`` on segment ``seg`` (between vertices
    ``seg`` and ``seg + 1``).

    Returns (steps, chord_sum, last_chord, lat, lon, lat_prev, lon_prev): the last
    station and the one before it.
    """
    n = vlat.size
    r2 = (d / R) * (d / R)
    steps = 0
    chord_sum = 0.0
    last_chord = 0.0
    qlat = plat
    qlon = plon
    while True:
        pcos = math.cos(plat)
        # --- first vertex outside the divider circle, walking away from the station.
        # A vertex nearer along the path than the current shortfall cannot be outside
        # (chord <= path length), so it is skipped without evaluation.
        k = seg + 1 if direction > 0 else seg
        outside = False
        fk = 0.0
        while True:
            fk = _haversine_c(plat, plon, pcos, vlat[k], vlon[k], R) - d
            if fk > 0.0:
                outside = True
                break
            if direction > 0:
                if k == n - 1:
                    break
                target = tk[k] - fk
                k += 1
                while k < n - 1 and tk[k] < target:
                    k += 1
            else:
                if k == 0:
                    break
                target = tk[k] + fk
                k -= 1
                while k > 0 and tk[k] > target:
                    k -= 1
        if not outside:             # the end of the line is inside the circle
            break
        # --- the crossing lies on the segment that ends at vertex k. Its inner end is
        # the station itself (same segment) or the previous vertex.
        sg = k - 1 if direction > 0 else k
        if sg == seg:
            elat = plat
            elon = plon
            fe = -d
        else:
            e = k - direction
            elat = vlat[e]
            elon = vlon[e]
            fe = _haversine_c(plat, plon, pcos, elat, elon, R) - d
        dlat = vlat[k] - elat
        dlon = vlon[k] - elon
        # First guess: circle-segment intersection in the local plane of the station.
        ex = (elon - plon) * pcos
        ey = elat - plat
        dx = dlon * pcos
        dy = dlat
        qa = dx * dx + dy * dy
        qb = ex * dx + ey * dy
        qc = ex * ex + ey * ey - r2
        disc = qb * qb - qa * qc
        w = 0.5
        if disc > 0.0:
            w = (-qb + math.sqrt(disc)) / qa
        if not (0.0 < w < 1.0):
            w = 0.5
        # Refinement on the exact great-circle distance: Illinois on w in [0, 1].
        seglen = tk[sg + 1] - tk[sg]
        a = 0.0
        fa = fe
        b = 1.0
        fb = fk
        side = 0
        f = 0.0
        slat = elat
        slon = elon
        for _ in range(max_iter):
            slat = elat + w * dlat
            slon = elon + w * dlon
            f = _haversine_c(plat, plon, pcos, slat, slon, R) - d
            if abs(f) <= ftol:
                break
            if f > 0.0:
                b = w
                fb = f
                if side == 1:
                    fa *= 0.5
                side = 1
            else:
                a = w
                fa = f
                if side == -1:
                    fb *= 0.5
                side = -1
            if (b - a) * seglen <= xtol:
                break
            w = b - fb * (b - a) / (fb - fa)
            if not (a < w < b):
                w = 0.5 * (a + b)
        chord = d + f
        steps += 1
        chord_sum += chord
        last_chord = chord
        qlat = plat
        qlon = plon
        plat = slat
        plon = slon
        seg = sg
    return steps, chord_sum, last_chord, plat, plon, qlat, qlon


@nb.njit(cache=True)
def ys_walk_polyline(lat0, lon0, seg0, direction, d, ftol, xtol, max_iter, R, vlat, vlon, tk):
    n = lat0.size
    steps = np.empty(n, np.int64)
    csum = np.empty(n)
    last = np.empty(n)
    lat_end = np.empty(n)
    lon_end = np.empty(n)
    lat_prev = np.empty(n)
    lon_prev = np.empty(n)
    for w in range(n):
        st, cs_, la, a, b, c, e = _ys_polyline_one(
            lat0[w], lon0[w], seg0[w], direction, d, ftol, xtol, max_iter, R, vlat, vlon, tk
        )
        steps[w] = st
        csum[w] = cs_
        last[w] = la
        lat_end[w] = a
        lon_end[w] = b
        lat_prev[w] = c
        lon_prev[w] = e
    return steps, csum, last, lat_end, lon_end, lat_prev, lon_prev

# ------------------------------ closed-loop yardstick (periodic polyline)
@nb.njit(cache=True)
def _ys_polyline_closed_one(plat0, plon0, t0, seg0, direction, d, ftol, xtol,
                             max_iter, R, vlat, vlon, tk):
    """One closed polygon lap, in + or - orientation. Return (L, n_points).

    The unwrapped chordal t-axis runs from t0 to t0 + direction*T.
    The final endpoint is the *start station*, not a fixed input vertex.
    Every vertex is visited at most once; crossings on partial segments are
    bracketed against the exact great-circle chord.
    """
    N = len(tk) - 1
    T = tk[-1]
    t_limit = t0 + direction * T
    tcur = t0
    plat, plon = plat0, plon0
    steps = 0
    chord_sum = 0.0
    k = seg0 + 1 if direction > 0 else seg0
    tol_t = 8.0 * 2.220446049250313e-16 * T
    while True:
        inner_t, inner_lat, inner_lon = tcur, plat, plon
        cos_p = math.cos(plat)
        found = False
        while True:
            cycle = k // N
            local = k % N
            vt = cycle * T + tk[local]
            terminal = (direction * (vt - t_limit)) >= 0
            if terminal:
                vt = t_limit
                vla, vlo = plat0, plon0
            else:
                vla, vlo = vlat[local], vlon[local]
            # A physical longitude may jump by 2*pi at the file seam. Unwrap the
            # candidate near the previous point before interpolating a segment.
            vlo = inner_lon + (vlo - inner_lon + math.pi) % (2.0 * math.pi) - math.pi
            fv = _haversine_c(plat, plon, cos_p, vla, vlo, R) - d
            if fv >= -ftol:
                # First exit from the geodesic ball occurs on this segment.
                if abs(fv) <= ftol:
                    tc, slat, slon, f = vt, vla, vlo, fv
                else:
                    a, b = 0.0, 1.0
                    fa = _haversine_c(plat, plon, cos_p, inner_lat, inner_lon, R) - d
                    fb = fv
                    side = 0
                    w = -fa / (fb - fa)
                    if not (0.0 < w < 1.0):
                        w = 0.5
                    dt = vt - inner_t
                    for _ in range(max_iter):
                        w_eval = w
                        slat = inner_lat + w * (vla - inner_lat)
                        slon = inner_lon + w * (vlo - inner_lon)
                        f = _haversine_c(plat, plon, cos_p, slat, slon, R) - d
                        if abs(f) <= ftol:
                            break
                        if f > 0.0:
                            b, fb = w, f
                            if side == 1:
                                fa *= 0.5
                            side = 1
                        else:
                            a, fa = w, f
                            if side == -1:
                                fb *= 0.5
                            side = -1
                        if (b - a) * abs(dt) <= xtol:
                            break
                        w = b - fb * (b - a) / (fb - fa)
                        if not (a < w < b):
                            w = 0.5 * (a + b)
                    tc = inner_t + w_eval * dt
                if direction * (tc - tcur) <= 0.0:
                    raise RuntimeError("periodic YS failed to advance")
                steps += 1
                chord_sum += d + f
                tcur, plat, plon = tc, slat, slon
                if (direction * (t_limit - tcur) <= tol_t):
                    # At the start again; closure chord is zero.
                    return chord_sum, steps + 1
                k = k + direction if abs(tc - vt) <= tol_t else k
                found = True
                break
            if terminal:
                break
            inner_t, inner_lat, inner_lon = vt, vla, vlo
            k += direction
        if not found:
            break
    if steps == 0:
        raise ValueError("ruler is too large: no complete yardstick step on a full lap")
    closure = _haversine_c(plat, plon, math.cos(plat), plat0, plon0, R)
    return chord_sum + closure, steps + (1 if closure <= ftol else 2)

@nb.njit(cache=True)
def ys_walk_polyline_closed(lat0, lon0, t0, seg0, direction, d, ftol, xtol,
                            max_iter, R, vlat, vlon, tk):
    n = lat0.size
    lengths = np.empty(n)
    n_points = np.empty(n, dtype=np.int64)
    for i in range(n):
        lengths[i], n_points[i] = _ys_polyline_closed_one(
            lat0[i], lon0[i], t0[i], seg0[i], direction[i], d, ftol,
            xtol, max_iter, R, vlat, vlon, tk,
        )
    return lengths, n_points
