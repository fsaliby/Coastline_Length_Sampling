"""Monte-Carlo estimators of coastline length.

All estimators operate in the arc-length parameter of a
:class:`~coastline_geometry.CoastlineModel` and come in two engines with the same
results (up to round-off): ``"numpy"`` (vectorised, no extra dependency) and
``"numba"`` (compiled loops from :mod:`coastline_kernels`; faster, no temporaries).

``random_points``
    Open: ``n_p`` stations with endpoints pinned, chord-to-chord. Closed:
    ``n_p-1`` distinct uniform stations joined in a cycle (the repeated start is
    station ``n_p``). One call per ``n_p``; realisations are independent across calls.

``random_points_nested``
    The whole ``n_p`` sweep from a single set of draws per realisation: open
    curves use endpoints + the first ``n-2`` draws; closed curves use the first
    ``n-1`` draws, connected cyclically.
    Each ``P(L | n)`` is exactly the same distribution as with ``random_points``,
    but samples at different ``n`` from the same realisation are *dependent*
    (common random numbers). Cost is O(n_max) per realisation instead of
    O(sum of n).

``yardstick``
    Open: a chord of length ``d`` walked from a random seed station in both
    directions with endpoint policies. Closed: one full periodic lap in a fair
    random direction; each station is the *first* chord crossing, and one
    closing chord connects the final station to the random start. Closed YS
    requires a polyline and end_policy='keep'.

    * **march** - the chord can grow by at most ``lip * ds`` when the far end moves
      by ``ds`` (``lip = model.arc_lipschitz``), so from a point where the chord falls
      short by ``g`` the next ``g / lip`` of arc cannot hold a crossing and is skipped
      without evaluation. The march starts at ``d / lip`` for the same reason. Steps
      are floored at ``hmin = 1e-3 * d``, which is the resolution below which a brief
      excursion of the chord above ``d`` could go unnoticed (the previous fixed scan
      had 1/640 * d);
    * **bracket** - once the chord exceeds ``d``, Illinois (modified regula falsi)
      on the bracket, to ``|chord - d| <= 1e-10 * d``.

    The interior length is the sum of the chords actually obtained, not
    ``n_steps * d``, so the result does not depend on the root tolerance.

    ``walker="polyline"`` (piecewise-linear curve only) replaces the march by a
    geometric argument. The divider circle is convex and the curve is made of
    straight segments, so

    * a segment with both ends inside the circle lies inside it;
    * a segment going from inside to outside crosses the circle exactly once.

    The first crossing is therefore on the segment that ends at the first *vertex*
    outside the circle: only vertices are tested, and those nearer along the path
    than the current shortfall are skipped (chord <= path length). There is no
    ``hmin`` floor, no Lipschitz constant, and no excursion that can go unnoticed
    between two trials. On that segment the station solves ``chord = d``; with
    segments straight in (lat, lon) and great-circle distances this has no closed
    form, so the planar circle-segment intersection is used as first guess and
    refined by Illinois on the exact distance, to the same tolerance as the march.

    Validity. A segment straight in (lat, lon) is not a geodesic: it turns with
    geodesic curvature ``sin(azimuth)**3 * tan(lat) / R``, at most ``tan|lat| / R``. The
    two statements above hold as long as this is below the curvature of the divider
    circle, ``cot(d / R) / R``, i.e. for

        d < R * (pi / 2 - max |lat|)            (:func:`polyline_ruler_limit`)

    the distance from the curve to the nearest pole (7 450 km at 23 deg, 334 km at
    87 deg). Longer rulers are refused; use ``walker="march"`` for them.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from coastline_geometry import CoastlineModel, haversine

__all__ = [
    "EndPolicy", "Engine", "YsWalker", "random_points", "random_points_nested",
    "nested_block_seed", "yardstick", "polyline_ruler_limit", "closed_ruler_limit",
]

EndPolicy = Literal["merge", "keep", "drop"]
Engine = Literal["numpy", "numba"]
UniformIn = Literal["arc", "chordal"]
YsWalker = Literal["march", "polyline"]

_LIP_MARGIN = 1.01          # safety factor on the sampled Lipschitz estimate
_MARCH_FLOOR = 1e-3         # hmin / d
_ROOT_FTOL = 1e-8          # |chord - d| / d at acceptance
_ROOT_MAX_ITERS = 60

NESTED_BLOCK = 256          # realisations per independently seeded nested block


def _kernels():
    import coastline_kernels  # deferred: numba is optional

    return coastline_kernels


def _tables(model: CoastlineModel):
    """Kernel tables, built once per model and cached on it."""
    tab = getattr(model, "_kernel_tables", None)
    if tab is None:
        tab = _kernels().KernelTables(model)
        model._kernel_tables = tab
    return tab


def _sampling_domain(model: CoastlineModel, uniform_in: UniformIn):
    if uniform_in == "arc":
        return 0.0, model.length, model.points, True
    if uniform_in == "chordal":
        lo, hi = model.t_span
        return lo, hi, model.points_of_t, False
    raise ValueError(f"unknown uniform_in {uniform_in!r}")


def _unit_vectors(lat: NDArray[np.float64], lon: NDArray[np.float64]) -> NDArray[np.float64]:
    cl = np.cos(lat)
    return np.stack([cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)], axis=-1)


def _chord_uv(a: NDArray[np.float64], b: NDArray[np.float64], radius: float) -> NDArray[np.float64]:
    """Great-circle distance between unit vectors (same conditioning as haversine)."""
    half = 0.5 * np.sqrt(np.sum((a - b) ** 2, axis=-1))
    return 2.0 * radius * np.arcsin(np.minimum(half, 1.0))


# ================================================================ random points
def random_points(
    model: CoastlineModel,
    n_points: int,
    n_realisations: int,
    rng: np.random.Generator,
    uniform_in: UniformIn = "arc",
    max_elements: int = 1_048_576,
    engine: Engine = "numpy",
) -> NDArray[np.float64]:
    if model.topology == "closed":
        return _random_points_closed(
            model, n_points, n_realisations, rng, uniform_in, max_elements, engine,
        )
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    if n_realisations < 1:
        raise ValueError("n_realisations must be >= 1")

    lo_p, hi_p, to_points, use_map = _sampling_domain(model, uniform_in)
    span = hi_p - lo_p

    out = np.empty(n_realisations, dtype=float)
    if n_points == 2:
        lat, lon = to_points(np.array([lo_p, hi_p]))
        out[:] = haversine(lat[0], lon[0], lat[1], lon[1], model.radius)
        return out

    if engine == "numba":
        tab = _tables(model)
        kernel = _kernels().rp_independent
    elif engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")

    # Draws are taken in row-major blocks, so results do not depend on the block size.
    chunk = max(1, min(n_realisations, int(max_elements // n_points)))
    for lo in range(0, n_realisations, chunk):
        hi = min(lo + chunk, n_realisations)
        m = hi - lo
        p = np.empty((m, n_points), dtype=float)
        p[:, 0] = lo_p
        p[:, -1] = hi_p
        p[:, 1:-1] = rng.random((m, n_points - 2)) * span + lo_p
        p[:, 1:-1].sort(axis=1)

        if engine == "numba":
            out[lo:hi] = kernel(p, *tab.rp_args(use_map), tab.radius)
        else:
            lat, lon = to_points(p)
            out[lo:hi] = haversine(
                lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:], model.radius
            ).sum(axis=1)
    return out


def _random_points_closed(
    model: CoastlineModel, n_points: int, n_realisations: int,
    rng: np.random.Generator, uniform_in: UniformIn,
    max_elements: int, engine: Engine,
) -> NDArray[np.float64]:
    """Periodic RP: n_points-1 independent stations and the closing chord.

    The repeated start on the last polygon vertex is counted in n_points, so
    x=n_points-1 remains the number of measured chords in both topologies.
    """
    if n_points < 3 or n_realisations < 1:
        raise ValueError("closed RP requires n_points >= 3 and n_realisations >= 1")
    lo_p, hi_p, to_points, use_map = _sampling_domain(model, uniform_in)
    span = hi_p - lo_p
    n_unique = n_points - 1
    out = np.empty(n_realisations)
    if engine == "numba":
        tab = _tables(model)
        kernel = _kernels().rp_independent_closed
    elif engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")
    chunk = max(1, min(n_realisations, int(max_elements // n_unique)))
    for lo in range(0, n_realisations, chunk):
        hi = min(lo + chunk, n_realisations)
        p = rng.random((hi - lo, n_unique)) * span + lo_p
        p.sort(axis=1)
        if engine == "numba":
            out[lo:hi] = kernel(p, *tab.rp_args(use_map), tab.radius)
        else:
            lat, lon = to_points(p)
            out[lo:hi] = (
                haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:], model.radius)
                .sum(axis=1)
                + haversine(lat[:, -1], lon[:, -1], lat[:, 0], lon[:, 0], model.radius)
            )
    return out


def nested_block_seed(seed: int, block: int) -> np.random.SeedSequence:
    """Seed of nested block ``block``: depends only on (seed, block index)."""
    return np.random.SeedSequence([int(seed), 0x4E53, int(block)])


def random_points_nested(
    model: CoastlineModel,
    resolutions: Sequence[int],
    n_realisations: int,
    rng: np.random.Generator,
    uniform_in: UniformIn = "arc",
    max_elements: int = 1_048_576,
    engine: Engine = "numpy",
) -> NDArray[np.float64]:
    """Lengths for every requested ``n`` from one nested draw per realisation.

    Returns an array of shape ``(n_realisations, len(resolutions))``. Column ``k``
    is an exact sample of ``P(L | resolutions[k])``; columns are dependent.
    """
    if model.topology == "closed":
        return _random_points_nested_closed(
            model, resolutions, n_realisations, rng, uniform_in, max_elements, engine,
        )
    res = np.asarray(resolutions, dtype=np.int64)
    if res.ndim != 1 or res.size == 0 or res.min() < 2:
        raise ValueError("resolutions must be a non-empty list of integers >= 2")
    n_max = int(res.max())
    K = n_max - 2
    lo_p, hi_p, to_points, use_map = _sampling_domain(model, uniform_in)
    span = hi_p - lo_p
    R = model.radius

    cols = np.full(n_max + 1, -1, dtype=np.int64)
    cols[res] = np.arange(res.size)
    out = np.empty((n_realisations, res.size + 1))  # last column: total at n = 2

    lat_e, lon_e = to_points(np.array([lo_p, hi_p]))
    direct_n2 = float(haversine(lat_e[0], lon_e[0], lat_e[1], lon_e[1], R))

    if K == 0:
        out[:, :] = direct_n2
        return out[:, :-1]

    if engine == "numba":
        tab = _tables(model)
        kernel = _kernels().rp_nested
    elif engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")

    # Rows per pass, from the number of stations per row; draws stay row-major, so
    # the block size does not change the result.
    chunk = max(1, min(n_realisations, int(max_elements // max(n_max, 1))))
    for lo in range(0, n_realisations, chunk):
        hi = min(lo + chunk, n_realisations)
        m = hi - lo
        U = rng.random((m, K)) * span + lo_p          # insertion order
        if engine == "numba":
            kernel(U, lo_p, hi_p, *tab.rp_args(use_map), R, cols, out[lo:hi])
            continue

        rows = np.arange(m)
        order = np.argsort(U, axis=1)
        rank = np.empty((m, K), dtype=np.int64)
        np.put_along_axis(rank, order, np.broadcast_to(np.arange(1, K + 1), (m, K)), axis=1)
        ps = np.empty((m, n_max))
        ps[:, 0] = lo_p
        ps[:, -1] = hi_p
        ps[:, 1:-1] = np.take_along_axis(U, order, axis=1)
        del order
        V = _unit_vectors(*to_points(ps))              # (m, N, 3)
        del ps
        rc = np.empty((m, n_max))                      # chord to current successor
        rc[:, :-1] = _chord_uv(V[:, :-1], V[:, 1:], R)
        tot = rc[:, :-1].sum(axis=1)
        nxt = np.broadcast_to(np.arange(1, n_max + 1), (m, n_max)).copy()
        prv = np.broadcast_to(np.arange(-1, n_max - 1), (m, n_max)).copy()
        if cols[n_max] >= 0:
            out[lo:hi, cols[n_max]] = tot
        for k in range(K - 1, -1, -1):
            x = rank[:, k]
            a = prv[rows, x]
            b = nxt[rows, x]
            new = _chord_uv(V[rows, a], V[rows, b], R)
            tot += new - rc[rows, a] - rc[rows, x]
            rc[rows, a] = new
            nxt[rows, a] = b
            prv[rows, b] = a
            if cols[k + 2] >= 0:
                out[lo:hi, cols[k + 2]] = tot
        out[lo:hi, -1] = tot

    drift = np.max(np.abs(out[:, -1] - direct_n2))
    if drift > 1e-9 * max(model.length, 1.0):
        raise RuntimeError(f"nested sweep round-off drift {drift:.3e} km exceeds tolerance")
    return out[:, :-1]


def _random_points_nested_closed(
    model: CoastlineModel, resolutions: Sequence[int], n_realisations: int,
    rng: np.random.Generator, uniform_in: UniformIn, max_elements: int, engine: Engine,
) -> NDArray[np.float64]:
    """Periodic nested RP, removing sampled points in reverse insertion order.

    After every removal, two circular neighbours are reconnected. Both the
    saved marginals and the dependency between resolutions have the intended
    meaning; no vertex of the input ring is singled out.
    """
    res = np.asarray(resolutions, dtype=np.int64)
    if res.ndim != 1 or res.size == 0 or res.min() < 3 or n_realisations < 1:
        raise ValueError("closed nested RP requires n_points >= 3 and realisations >= 1")
    n_max = int(res.max())
    n_unique = n_max - 1
    cols = np.full(n_max + 1, -1, dtype=np.int64)
    cols[res] = np.arange(res.size)
    out = np.empty((n_realisations, res.size), dtype=float)
    lo_p, hi_p, to_points, use_map = _sampling_domain(model, uniform_in)
    span = hi_p - lo_p
    R = model.radius
    if engine == "numba":
        tab = _tables(model)
        kernel = _kernels().rp_nested_closed
    elif engine != "numpy":
        raise ValueError(f"unknown engine {engine!r}")

    chunk = max(1, min(n_realisations, int(max_elements // n_unique)))
    for lo in range(0, n_realisations, chunk):
        hi = min(lo + chunk, n_realisations)
        m = hi - lo
        U = rng.random((m, n_unique)) * span + lo_p
        if engine == "numba":
            kernel(U, *tab.rp_args(use_map), R, cols, out[lo:hi])
            continue

        rows = np.arange(m)
        order = np.argsort(U, axis=1)
        rank = np.argsort(order, axis=1)
        ps = np.take_along_axis(U, order, axis=1)
        V = _unit_vectors(*to_points(ps))
        del ps, order
        rc = _chord_uv(V, np.roll(V, -1, axis=1), R)
        tot = rc.sum(axis=1)
        nxt = np.broadcast_to((np.arange(n_unique) + 1) % n_unique, (m, n_unique)).copy()
        prv = np.broadcast_to((np.arange(n_unique) - 1) % n_unique, (m, n_unique)).copy()
        if cols[n_max] >= 0:
            out[lo:hi, cols[n_max]] = tot
        for k in range(n_unique - 1, 1, -1):
            x = rank[:, k]
            a = prv[rows, x]
            b = nxt[rows, x]
            new = _chord_uv(V[rows, a], V[rows, b], R)
            tot += new - rc[rows, a] - rc[rows, x]
            rc[rows, a] = new
            nxt[rows, a] = b
            prv[rows, b] = a
            if cols[k + 1] >= 0:
                out[lo:hi, cols[k + 1]] = tot
        # With only two stations left, the closed polygon has two identical chords.
        first = _unit_vectors(*to_points(U[:, :2]))
        direct_two = 2.0 * _chord_uv(first[:, 0], first[:, 1], R)
        drift = np.max(np.abs(tot - direct_two))
        if drift > 1e-9 * max(model.length, 1.0):
            raise RuntimeError(f"periodic nested RP drift {drift:.3e} km")
    return out


# ==================================================================== yardstick
def _walk_numpy(
    model: CoastlineModel,
    s0: NDArray[np.float64],
    direction: NDArray[np.int64],
    d: float,
    lip: float,
    ftol: float,
    xtol: float,
    hmin: float,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64]]:
    """All walkers advance asynchronously: one point evaluation per walker per pass.

    A walker that finishes a step starts the next one in the same pass, so the
    number of passes is set by the busiest walker's total evaluations, not by
    (steps x the slowest walker at each step).
    """
    L = model.length
    R = model.radius
    n = s0.size
    base = s0.astype(float).copy()
    lat_b, lon_b = model.points(base)
    steps = np.zeros(n, dtype=np.int64)
    csum = np.zeros(n)
    last = np.zeros(n)
    s_prev = base.copy()

    phase = np.zeros(n, dtype=np.int8)       # 0: march, 1: bracket
    lo = np.zeros(n)
    flo = np.full(n, -d)
    a = np.zeros(n)
    fa = np.zeros(n)
    b = np.zeros(n)
    fb = np.zeros(n)
    side = np.zeros(n, dtype=np.int8)
    iters = np.zeros(n, dtype=np.int64)
    dmax = np.where(direction > 0, L - base, base)
    trial = np.minimum(d / lip, dmax)
    active = lip * dmax >= d

    while True:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        tr = trial[idx]
        s_t = base[idx] + direction[idx] * tr
        lat, lon = model.points(s_t)
        f = haversine(lat_b[idx], lon_b[idx], lat, lon, R) - d
        ph = phase[idx]
        accept = np.abs(f) <= ftol

        # ---- march walkers
        march = (ph == 0) & ~accept
        cross = march & (f > 0.0)
        short = march & ~cross
        at_end = short & (tr >= dmax[idx])
        go = short & ~at_end
        active[idx[at_end]] = False
        if go.any():
            i = idx[go]
            lo[i] = tr[go]
            flo[i] = f[go]
            trial[i] = np.minimum(tr[go] + np.maximum(-f[go] / lip, hmin), dmax[i])
        if cross.any():
            i = idx[cross]
            a[i] = lo[i]
            fa[i] = flo[i]
            b[i] = tr[cross]
            fb[i] = f[cross]
            side[i] = 0
            iters[i] = 0
            phase[i] = 1

        # ---- bracket walkers (Illinois)
        brk = (ph == 1) & ~accept
        if brk.any():
            i = idx[brk]
            fc = f[brk]
            c = tr[brk]
            right = fc > 0.0
            sd = side[i]
            new_fa = np.where(right, np.where(sd == 1, 0.5 * fa[i], fa[i]), fc)
            new_fb = np.where(right, fc, np.where(sd == -1, 0.5 * fb[i], fb[i]))
            a[i] = np.where(right, a[i], c)
            b[i] = np.where(right, c, b[i])
            fa[i] = new_fa
            fb[i] = new_fb
            side[i] = np.where(right, 1, -1)
            iters[i] += 1
            accept[brk] = ((b[i] - a[i]) <= xtol) | (iters[i] >= _ROOT_MAX_ITERS)

        # ---- next trial for every walker still bracketing (incl. those just crossed)
        nxt_br = (phase[idx] == 1) & ~accept
        if nxt_br.any():
            i = idx[nxt_br]
            ai, bi = a[i], b[i]
            c = bi - fb[i] * (bi - ai) / (fb[i] - fa[i])
            bad = ~((c > ai) & (c < bi))
            c[bad] = 0.5 * (ai[bad] + bi[bad])
            trial[i] = c

        # ---- accepted stations: record the step, start the next one
        if accept.any():
            i = idx[accept]
            chord = d + f[accept]
            steps[i] += 1
            csum[i] += chord
            last[i] = chord
            s_prev[i] = base[i]
            base[i] = s_t[accept]
            lat_b[i] = lat[accept]
            lon_b[i] = lon[accept]
            phase[i] = 0
            lo[i] = 0.0
            flo[i] = -d
            dmax[i] = np.where(direction[i] > 0, L - base[i], base[i])
            trial[i] = np.minimum(d / lip, dmax[i])
            active[i] = lip * dmax[i] >= d

    return steps, csum, last, base, s_prev


def _haversine_c(lat1, lon1, cos1, lat2, lon2, radius):
    """:func:`~coastline_geometry.haversine` with ``cos(lat1)`` supplied."""
    a = np.sin(0.5 * (lat2 - lat1))
    b = np.sin(0.5 * (lon2 - lon1))
    h = a * a + cos1 * np.cos(lat2) * (b * b)
    return 2.0 * radius * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def _walk_polyline_numpy(
    vlat: NDArray[np.float64],
    vlon: NDArray[np.float64],
    tk: NDArray[np.float64],
    radius: float,
    lat0: NDArray[np.float64],
    lon0: NDArray[np.float64],
    seg0: NDArray[np.int64],
    direction: NDArray[np.int64],
    d: float,
    ftol: float,
    xtol: float,
) -> Tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64],
           NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Polyline walkers, advanced asynchronously: one distance per walker per pass.

    Same algorithm, operation for operation, as ``coastline_kernels._ys_polyline_one``.
    Each walker is in one of three states, which says what its next distance is for:
    0 - testing a vertex; 1 - measuring the inner end of the crossing segment;
    2 - refining the crossing on that segment.
    """
    n_v = vlat.size
    n = lat0.size
    fwd = direction > 0
    r2 = (d / radius) * (d / radius)

    plat = lat0.astype(float).copy()
    plon = lon0.astype(float).copy()
    pcos = np.cos(plat)
    seg = seg0.astype(np.int64).copy()
    qlat = plat.copy()
    qlon = plon.copy()
    steps = np.zeros(n, dtype=np.int64)
    csum = np.zeros(n)
    last = np.zeros(n)

    state = np.zeros(n, dtype=np.int8)
    k = np.where(fwd, seg + 1, seg)          # vertex under test / outer end of the crossing
    xlat = vlat[k]                           # point whose distance is taken in the next pass
    xlon = vlon[k]
    sg = np.zeros(n, dtype=np.int64)         # segment holding the crossing
    fk = np.zeros(n)                         # chord - d at vertex k
    elat = np.zeros(n)                       # inner end of the crossing segment
    elon = np.zeros(n)
    dlat = np.zeros(n)
    dlon = np.zeros(n)
    seglen = np.zeros(n)
    a = np.zeros(n)
    fa = np.zeros(n)
    b = np.zeros(n)
    fb = np.zeros(n)
    w = np.zeros(n)
    side = np.zeros(n, dtype=np.int8)
    iters = np.zeros(n, dtype=np.int64)
    active = np.ones(n, dtype=bool)

    while True:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        f = _haversine_c(plat[idx], plon[idx], pcos[idx], xlat[idx], xlon[idx], radius) - d
        st = state[idx]
        accept = np.zeros(idx.size, dtype=bool)
        begin = np.zeros(idx.size, dtype=bool)        # root search starts in this pass

        # ---- state 0: vertex test
        vertex = st == 0
        inside = vertex & ~(f > 0.0)
        if inside.any():
            i = idx[inside]
            ki = k[i]
            at_end = np.where(fwd[i], ki == n_v - 1, ki == 0)
            active[i[at_end]] = False                 # the end of the line is inside the circle
            i, ki, fi = i[~at_end], ki[~at_end], f[inside][~at_end]
            if i.size:
                # skip the vertices nearer along the path than the shortfall -f
                up = fwd[i]
                target = np.where(up, tk[ki] - fi, tk[ki] + fi)
                k_up = np.minimum(
                    np.maximum(np.searchsorted(tk, target, side="left"), ki + 1), n_v - 1
                )
                k_dn = np.maximum(
                    np.minimum(np.searchsorted(tk, target, side="right") - 1, ki - 1), 0
                )
                kn = np.where(up, k_up, k_dn)
                k[i] = kn
                xlat[i] = vlat[kn]
                xlon[i] = vlon[kn]
        outside = vertex & (f > 0.0)
        if outside.any():
            pos = np.flatnonzero(outside)
            i = idx[pos]
            ki = k[i]
            sgi = np.where(fwd[i], ki - 1, ki)
            sg[i] = sgi
            fk[i] = f[pos]
            own = sgi == seg[i]                       # crossing on the station's own segment
            io = i[own]
            elat[io] = plat[io]
            elon[io] = plon[io]
            fa[io] = -d
            begin[pos[own]] = True
            iv = i[~own]                              # inner end = previous vertex: measure it
            e = ki[~own] - direction[iv]
            elat[iv] = vlat[e]
            elon[iv] = vlon[e]
            xlat[iv] = vlat[e]
            xlon[iv] = vlon[e]
            state[iv] = 1

        # ---- state 1: distance of the inner end
        inner = st == 1
        if inner.any():
            fa[idx[inner]] = f[inner]
            begin |= inner

        # ---- start of a root search: planar first guess
        if begin.any():
            i = idx[begin]
            dla = vlat[k[i]] - elat[i]
            dlo = vlon[k[i]] - elon[i]
            ex = (elon[i] - plon[i]) * pcos[i]
            ey = elat[i] - plat[i]
            dx = dlo * pcos[i]
            dy = dla
            qa = dx * dx + dy * dy
            qb = ex * dx + ey * dy
            qc = ex * ex + ey * ey - r2
            disc = qb * qb - qa * qc
            w0 = np.full(i.size, 0.5)
            ok = disc > 0.0
            w0[ok] = (-qb[ok] + np.sqrt(disc[ok])) / qa[ok]
            w0[~((w0 > 0.0) & (w0 < 1.0))] = 0.5
            dlat[i] = dla
            dlon[i] = dlo
            seglen[i] = tk[sg[i] + 1] - tk[sg[i]]
            a[i] = 0.0
            b[i] = 1.0
            fb[i] = fk[i]
            side[i] = 0
            iters[i] = 0
            w[i] = w0
            xlat[i] = elat[i] + w0 * dla
            xlon[i] = elon[i] + w0 * dlo
            state[i] = 2

        # ---- state 2: Illinois on the exact distance
        root = st == 2
        if root.any():
            i = idx[root]
            fr = f[root]
            stop = np.abs(fr) <= ftol
            iters[i] += 1
            upd = ~stop
            if upd.any():
                iu = i[upd]
                fu = fr[upd]
                wu = w[iu]
                right = fu > 0.0
                sd = side[iu]
                new_fa = np.where(right, np.where(sd == 1, 0.5 * fa[iu], fa[iu]), fu)
                new_fb = np.where(right, fu, np.where(sd == -1, 0.5 * fb[iu], fb[iu]))
                a[iu] = np.where(right, a[iu], wu)
                b[iu] = np.where(right, wu, b[iu])
                fa[iu] = new_fa
                fb[iu] = new_fb
                side[iu] = np.where(right, 1, -1)
                stop[upd] = ((b[iu] - a[iu]) * seglen[iu] <= xtol) | (iters[iu] >= _ROOT_MAX_ITERS)
            go = ~stop
            if go.any():
                ic = i[go]
                ai, bi = a[ic], b[ic]
                c = bi - fb[ic] * (bi - ai) / (fb[ic] - fa[ic])
                bad = ~((c > ai) & (c < bi))
                c[bad] = 0.5 * (ai[bad] + bi[bad])
                w[ic] = c
                xlat[ic] = elat[ic] + c * dlat[ic]
                xlon[ic] = elon[ic] + c * dlon[ic]
            accept[root] = stop

        # ---- accepted stations: record the step, start the next one
        if accept.any():
            i = idx[accept]
            chord = d + f[accept]
            steps[i] += 1
            csum[i] += chord
            last[i] = chord
            qlat[i] = plat[i]
            qlon[i] = plon[i]
            plat[i] = xlat[i]
            plon[i] = xlon[i]
            pcos[i] = np.cos(plat[i])
            seg[i] = sg[i]
            kn = np.where(fwd[i], seg[i] + 1, seg[i])
            k[i] = kn
            xlat[i] = vlat[kn]
            xlon[i] = vlon[kn]
            state[i] = 0

    return steps, csum, last, plat, plon, qlat, qlon


def polyline_ruler_limit(model: CoastlineModel) -> float:
    """Largest ruler (exclusive, km) for which ``walker="polyline"`` is exact on this
    curve: the distance from its highest-latitude vertex to the nearest pole (see the
    module docstring)."""
    return float(model.radius * (0.5 * np.pi - np.max(np.abs(model.lat_rad))))


def closed_ruler_limit(model: CoastlineModel, required: float | None = None) -> float:
    """Certified *lower bound* on the ruler size allowed for all closed-YS seeds.

    Let e(P) = max_{Q on the ring} distance(P,Q). A full lap can take a step
    from every seed when d < min_P e(P). We need only a sufficient lower bound
    on that critical size, not its exact value.

    For a set of witness vertices W, g(P) = max_{Q in W} distance(P,Q) <= e(P).
    Since g is 1-Lipschitz with respect to arc length, its values at both ends
    of any interval of length h imply the certified bound

                  g(P) >= (g(left) + g(right) - h) / 2.

    Begin with 16 farthest-first witness vertices (O(N) for a fixed witness
    count). If ``required`` does not pass this cheap test, progressively add
    witnesses, then subdivide *only* intervals not yet certified. On small
    contours we can use all vertices, but large inputs never incur an
    unconditional O(N**2) pairwise-distance calculation.

    A failure to certify d is NOT proof that some seed cannot take a step.
    The returned bound can underestimate the true threshold. It is cached
    on the model; asking later about a larger ruler may tighten the bound.
    """
    if model.topology != "closed":
        raise ValueError("closed_ruler_limit applies to closed curves only")
    if required is not None and (not np.isfinite(required) or required <= 0.0):
        raise ValueError("required ruler must be finite and positive")

    cached = getattr(model, "_closed_ruler_limit", None)
    if cached is not None and (required is None or required < cached):
        return float(cached)

    lat, lon = model.lat_rad[:-1], model.lon_rad[:-1]
    n = lat.size
    R = model.radius
    # The closing vertex appears only once in the witness set; s includes it.
    s = np.asarray(model.s_of_t(model.t_knots), dtype=float)
    nearest = np.full(n, np.inf)
    g = np.zeros(n, dtype=float)
    witnesses: list[int] = []
    chosen = np.zeros(n, dtype=bool)

    def _bound(stations: NDArray[np.float64], vals: NDArray[np.float64]) -> float:
        return float(max(0.0, np.min(0.5 * (vals[:-1] + vals[1:] - np.diff(stations)))))

    # Farthest-first witnesses cover large and irregular rings much better than
    # choosing the first k file vertices, which would introduce an artificial seam.
    maximum = min(n, 256 if required is not None else 16)
    checkpoint = 16
    index = 0
    while len(witnesses) < maximum:
        witnesses.append(index)
        chosen[index] = True
        d = haversine(lat, lon, lat[index], lon[index], R)
        g = np.maximum(g, d)
        nearest = np.minimum(nearest, d)
        if len(witnesses) >= min(checkpoint, maximum):
            bound = _bound(s, np.r_[g, g[0]])
            if required is None or required < bound:
                model._closed_ruler_limit = bound
                return bound
            checkpoint = min(maximum, checkpoint * 4)
        if len(witnesses) == maximum:
            break
        nearest[chosen] = -np.inf
        index = int(np.argmax(nearest))

    bound = _bound(s, np.r_[g, g[0]])
    # Subdivide only cells for which the Lipschitz certificate is inconclusive.
    # A midpoint is just another evaluation of g; this tightens the guaranteed
    # lower bound without changing the measuring curve or its parameterization.
    stations = s.copy()
    values = np.r_[g, g[0]]
    wlat, wlon = lat[witnesses], lon[witnesses]
    for _ in range(9):
        if required is None or required < bound:
            break
        cell_bounds = 0.5 * (values[:-1] + values[1:] - np.diff(stations))
        bad = np.flatnonzero(cell_bounds <= required)
        if not bad.size:
            break
        # A very large coast with an inadmissible ruler must fail quickly, not
        # spend unlimited memory refining an impossible certificate.
        if bad.size > 50_000 or stations.size + bad.size > max(4 * n, 100_000):
            break
        midpoint = 0.5 * (stations[bad] + stations[bad + 1])
        movable = (midpoint > stations[bad]) & (midpoint < stations[bad + 1])
        bad, midpoint = bad[movable], midpoint[movable]
        if not bad.size:
            break
        lat_m, lon_m = model.points(midpoint)
        middle_values = np.zeros(midpoint.size)
        for wla, wlo in zip(wlat, wlon):
            middle_values = np.maximum(middle_values, haversine(lat_m, lon_m, wla, wlo, R))
        stations = np.insert(stations, bad + 1, midpoint)
        values = np.insert(values, bad + 1, middle_values)
        bound = _bound(stations, values)

    # Cache only a proven lower bound. The strict check below is deliberate:
    # equality is a boundary case, not an admissible ruler.
    model._closed_ruler_limit = bound
    return bound


# Both walker families return, for the forward and the backward walkers,
#     (steps, chord_sum, last_chord, station, previous_station)
# where a station is a tuple of arrays in whatever coordinates the walker uses, plus
# the two functions that measure the chord from such a station to the upper and to the
# lower end of the curve. The end-fragment policies are written once on top of that.
def _march_walkers(model, s, d, ftol, xtol, engine):
    """March + bracket walkers (any interpolant); stations are arc lengths."""
    total = model.length
    lip = model.arc_lipschitz * _LIP_MARGIN
    hmin = _MARCH_FLOOR * d
    m = s.size
    if engine == "numba":
        kern = _kernels().ys_walk
        kargs = _tables(model).map_args()
        fwd = kern(s, 1, d, lip, ftol, xtol, hmin, _ROOT_MAX_ITERS, total, model.radius, *kargs)
        bwd = kern(s, -1, d, lip, ftol, xtol, hmin, _ROOT_MAX_ITERS, total, model.radius, *kargs)
    else:
        both = np.concatenate([s, s])
        dirs = np.concatenate([np.ones(m, np.int64), -np.ones(m, np.int64)])
        out = _walk_numpy(model, both, dirs, d, lip, ftol, xtol, hmin)
        fwd = tuple(v[:m] for v in out)
        bwd = tuple(v[m:] for v in out)

    def pack(walk):
        steps, csum, last, s_end, s_prev = walk
        return steps, csum, last, (s_end,), (s_prev,)

    return (
        pack(fwd), pack(bwd),
        lambda s_: model.chord(s_, total),
        lambda s_: model.chord(0.0, s_),
    )


def _polyline_walkers(model, s, d, ftol, xtol, engine):
    """Segment-by-segment walkers (piecewise-linear curve); stations are (lat, lon)."""
    vlat, vlon, tk, radius = model.lat_rad, model.lon_rad, model.t_knots, model.radius
    # Seed stations are placed with the model (uniform in arc length) and then located
    # on the polyline: segment index from the chordal parameter.
    seg = np.clip(np.searchsorted(tk, model.t_of_s(s), side="right") - 1, 0, tk.size - 2)
    lat, lon = model.points(s)
    m = s.size
    if engine == "numba":
        kern = _kernels().ys_walk_polyline
        fwd = kern(lat, lon, seg, 1, d, ftol, xtol, _ROOT_MAX_ITERS, radius, vlat, vlon, tk)
        bwd = kern(lat, lon, seg, -1, d, ftol, xtol, _ROOT_MAX_ITERS, radius, vlat, vlon, tk)
    else:
        dirs = np.concatenate([np.ones(m, np.int64), -np.ones(m, np.int64)])
        out = _walk_polyline_numpy(
            vlat, vlon, tk, radius, np.concatenate([lat, lat]), np.concatenate([lon, lon]),
            np.concatenate([seg, seg]), dirs, d, ftol, xtol,
        )
        fwd = tuple(v[:m] for v in out)
        bwd = tuple(v[m:] for v in out)

    def pack(walk):
        steps, csum, last, lat_end, lon_end, lat_prev, lon_prev = walk
        return steps, csum, last, (lat_end, lon_end), (lat_prev, lon_prev)

    return (
        pack(fwd), pack(bwd),
        lambda la, lo: haversine(la, lo, vlat[-1], vlon[-1], radius),
        lambda la, lo: haversine(vlat[0], vlon[0], la, lo, radius),
    )


# ==================================================================== closed YS
def _ys_polyline_closed_one_python(plat0, plon0, t0, seg0, direction, d, ftol, xtol,
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


def _yardstick_closed(
    model: CoastlineModel, ruler: float, n_realisations: int | None,
    rng: np.random.Generator | None, end_policy: EndPolicy,
    max_elements: int, engine: Engine, s0: NDArray[np.float64] | None,
    walker: YsWalker, directions: NDArray[np.int64] | None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """One oriented full lap and exactly one closing chord, with no fixed seam.

    n_points counts all distinct measuring stations + the repeated start vertex.
    This matches x=n_points-1 measured chords in the RP and MC-fit pipeline.
    The direction is randomized independently from the seed station by the driver.
    """
    if model.kind != "linear" or walker != "polyline":
        raise ValueError("closed YS needs interpolation='linear' and walker='polyline'")
    if end_policy != "keep":
        raise ValueError("closed YS needs end_policy='keep' for a closed polygon")
    if not (np.isfinite(ruler) and ruler > 0.0):
        raise ValueError("ruler must be finite and positive")
    limit = polyline_ruler_limit(model)
    if not ruler < limit:
        raise ValueError(f"closed polyline ruler must be below {limit:g} km")
    # All or nothing: refuse a ruler for which some seed stations would see the whole
    # ring inside the divider circle, instead of failing on those seeds only.
    limit = closed_ruler_limit(model, required=ruler)
    if not ruler < limit:
        raise ValueError(
            f"closed YS: ruler {ruler:.6g} km is not below the certified safe "
            f"bound {limit:.6g} km. This is a sufficient bound, not the exact "
            "critical ruler; lower the ruler or refine the contour."
        )
    if engine not in ("numpy", "numba"):
        raise ValueError(f"unknown engine {engine!r}")
    if s0 is None:
        if rng is None or n_realisations is None or n_realisations < 1:
            raise ValueError("give either s0, or rng and n_realisations >= 1")
        s0 = rng.random(n_realisations) * model.length
    s0 = np.asarray(s0, dtype=float)
    if s0.ndim != 1 or np.any(~np.isfinite(s0)) or np.any((s0 < 0) | (s0 >= model.length)):
        raise ValueError("closed YS seed stations must lie in [0, L)")
    if directions is None:
        # Direct users can supply directions; in a seeded call they are drawn from
        # the same Generator after the stations. For campaigns, the driver derives
        # an independent stream shared across rulers when --ys-seeds=common.
        directions = (2 * rng.integers(0, 2, s0.size) - 1) if rng is not None else np.ones(s0.size, dtype=np.int64)
    directions = np.asarray(directions, dtype=np.int64)
    if directions.shape != s0.shape or np.any((directions != 1) & (directions != -1)):
        raise ValueError("directions must be a vector of +/-1, one per seed")
    t0 = model.t_of_s(s0)
    seg0 = np.clip(np.searchsorted(model.t_knots, t0, side="right") - 1,
                   0, model.t_knots.size - 2).astype(np.int64)
    lat0, lon0 = model.points(s0)
    R = model.radius
    ftol = _ROOT_FTOL * ruler
    xtol = max(1e-13 * ruler, 16.0 * np.finfo(float).eps * model.length)
    n = s0.size
    lengths = np.empty(n)
    n_points = np.empty(n, dtype=np.int64)
    chunk = max(1, min(n, max_elements))
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        if engine == "numba":
            l_, n_ = _kernels().ys_walk_polyline_closed(
                lat0[lo:hi], lon0[lo:hi], t0[lo:hi], seg0[lo:hi], directions[lo:hi],
                ruler, ftol, xtol, _ROOT_MAX_ITERS, R,
                model.lat_rad, model.lon_rad, model.t_knots,
            )
            lengths[lo:hi], n_points[lo:hi] = l_, n_
        else:
            for i in range(lo, hi):
                lengths[i], n_points[i] = _ys_polyline_closed_one_python(
                    lat0[i], lon0[i], t0[i], seg0[i], int(directions[i]),
                    ruler, ftol, xtol, _ROOT_MAX_ITERS, R,
                    model.lat_rad, model.lon_rad, model.t_knots,
                )
    return lengths, n_points


def yardstick(
    model: CoastlineModel,
    ruler: float,
    n_realisations: int | None = None,
    rng: np.random.Generator | None = None,
    end_policy: EndPolicy = "merge",
    max_elements: int = 1_048_576,
    engine: Engine = "numpy",
    s0: NDArray[np.float64] | None = None,
    walker: YsWalker = "march",
    directions: NDArray[np.int64] | None = None,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Yardstick lengths and station counts.

    Seed stations come either from ``rng`` (``n_realisations`` uniform draws in one
    call, as before) or are given directly as ``s0``; the walk itself is
    deterministic, so splitting ``s0`` across processes reproduces the serial result.

    ``walker`` selects how each station is found (see the module docstring):
    ``"march"`` works on open curves with any interpolant; ``"polyline"`` is
    the segment-by-segment walk and requires ``model.kind == "linear"``.
    Closed curves additionally require walker='polyline' and end_policy='keep'.
    ``directions`` optionally gives +1/-1 per realisation on closed curves.
    """
    if model.topology == "closed":
        return _yardstick_closed(model, ruler, n_realisations, rng, end_policy,
                                 max_elements, engine, s0, walker, directions)
    if ruler <= 0.0:
        raise ValueError("ruler must be positive")
    if engine not in ("numpy", "numba"):
        raise ValueError(f"unknown engine {engine!r}")
    if walker == "march":
        walkers = _march_walkers
    elif walker == "polyline":
        if model.kind != "linear":
            raise ValueError(
                "walker='polyline' walks straight segments and needs the piecewise-linear "
                f"curve (interpolation='linear'); this model is {model.kind!r}"
            )
        limit = polyline_ruler_limit(model)
        if not ruler < limit:
            raise ValueError(
                f"walker='polyline' is exact only for rulers below {limit:.6g} km on this "
                f"curve (its distance to the nearest pole); got {ruler:.6g} km. "
                "Use walker='march' for this ruler"
            )
        walkers = _polyline_walkers
    else:
        raise ValueError(f"unknown walker {walker!r}")
    total = model.length
    if s0 is None:
        if rng is None or n_realisations is None or n_realisations < 1:
            raise ValueError("give either s0, or rng and n_realisations >= 1")
        s0 = rng.random(n_realisations) * total
    s0 = np.asarray(s0, dtype=float)
    n_real = s0.size

    d = float(ruler)
    ftol = _ROOT_FTOL * d
    xtol = max(1e-13 * d, 16.0 * np.finfo(float).eps * total)

    lengths = np.empty(n_real, dtype=float)
    n_points = np.empty(n_real, dtype=np.int64)

    # Walker state is ~20 scalars per walker, 2 walkers per realisation, so each
    # state array has 2 * chunk elements.
    chunk = max(1, min(n_real, int(max_elements // 2)))

    for lo_i in range(0, n_real, chunk):
        hi_i = min(lo_i + chunk, n_real)
        fwd, bwd, to_hi, to_lo = walkers(model, s0[lo_i:hi_i], d, ftol, xtol, engine)
        fwd_steps, fwd_sum, fwd_last, at_hi, prev_hi = fwd
        bwd_steps, bwd_sum, bwd_last, at_lo, prev_lo = bwd

        n_steps = fwd_steps + bwd_steps
        interior = fwd_sum + bwd_sum
        frag_hi = to_hi(*at_hi)
        frag_lo = to_lo(*at_lo)

        # When the seed or the last full step is exactly at an endpoint,
        # the endpoint is already a measuring station. Do not count it twice.
        # Use the measured fragment (rather than a tolerance on s0) so that
        # this also handles steps landing exactly at either endpoint.
        zero_hi = frag_hi == 0.0
        zero_lo = frag_lo == 0.0

        if end_policy == "keep":
            total_len = interior + frag_hi + frag_lo
            n_stations = n_steps + 3 - zero_hi.astype(np.int64) - zero_lo.astype(np.int64)
        elif end_policy == "drop":
            total_len = interior
            n_stations = n_steps + 1
        elif end_policy == "merge":
            total_len = interior.copy()
            n_stations = n_steps + 3

            merge_hi = (frag_hi < 0.5 * d) & (fwd_steps > 0)
            keep_hi = ~merge_hi
            total_len[keep_hi] += frag_hi[keep_hi]
            if merge_hi.any():
                total_len[merge_hi] += (
                    to_hi(*(v[merge_hi] for v in prev_hi)) - fwd_last[merge_hi]
                )
                n_stations[merge_hi] -= 1

            merge_lo = (frag_lo < 0.5 * d) & (bwd_steps > 0)
            keep_lo = ~merge_lo
            total_len[keep_lo] += frag_lo[keep_lo]
            if merge_lo.any():
                total_len[merge_lo] += (
                    to_lo(*(v[merge_lo] for v in prev_lo)) - bwd_last[merge_lo]
                )
                n_stations[merge_lo] -= 1

            # Merging already removes the redundant endpoint-adjacent station.
            # If there was nothing to merge, avoid counting a zero fragment.
            n_stations -= (zero_hi & ~merge_hi).astype(np.int64)
            n_stations -= (zero_lo & ~merge_lo).astype(np.int64)
        else:
            raise ValueError(f"unknown end_policy {end_policy!r}")

        lengths[lo_i:hi_i] = total_len
        n_points[lo_i:hi_i] = n_stations

    return lengths, n_points
