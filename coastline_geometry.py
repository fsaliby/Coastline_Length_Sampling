"""Geometric model of a coastline.

The model has two distinct layers, and keeping them separate is the whole point
of this module:

1. **The curve.** A piecewise interpolant through the digitised (lat, lon)
   samples, parameterised by the *chordal* parameter ``t`` (cumulative
   great-circle distance along the input polyline). This layer defines the
   geometry and must not be altered by anything downstream.

2. **The parameterisation used for sampling.** A monotone map ``s <-> t``
   between true arc length ``s`` and the chordal parameter ``t``. Sampling
   uniformly in ``s`` is uniform *along the curve*; sampling uniformly in ``t``
   is not (it is biased against high-curvature stretches, where the spline arc
   overshoots the chord it was built from).

Arc length is obtained by Gauss-Legendre quadrature of the exact spherical
line element

    ds = R * sqrt( (dphi/dt)^2 + cos^2(phi) * (dlam/dt)^2 ) dt

evaluated on the analytic spline derivatives, on panels that never straddle a
knot. This gives the total length to ~1e-11 km.

The map ``t(s)`` is a piecewise cubic Hermite interpolant whose node values come
from that quadrature and whose node slopes are *exact*: ``dt/ds = 1 / |dP/dt|``
(one-sided at knots, so even the piecewise-linear curve, whose speed jumps at its
knots, is represented exactly there). Between nodes the map is only approximate,
and its error shows up as a *density distortion*

    q(s) = |dP/ds| = speed(t(s)) * t'(s)          (exactly 1 for a perfect map)

i.e. stations drawn uniformly in ``s`` fall with relative density ``q`` along the
true curve. The node grid is refined adaptively, splitting every cell where
``max |q - 1|`` exceeds ``map_tol``, until the whole curve passes. Refinement is
incremental: only the cells created by a split are integrated and tested again,
since a cell that is not split keeps its width in ``s`` and its end slopes, hence
its distortion. The curve itself (layer 1) is never touched by this.

Longitudes are unwrapped (``np.unwrap``) before fitting, so a line that crosses the
antimeridian stays continuous; longitudes may therefore leave [-180, 180] degrees.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
import warnings
from typing import Literal, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PPoly,
    make_interp_spline,
)

__all__ = [
    "EARTH_RADIUS_KM", "InputFormat", "InterpKind", "Topology", "haversine", "read_coordinates",
    "CoastlineModel",
]

EARTH_RADIUS_KM = 6371.0

InterpKind = Literal["cubic", "akima", "linear"]
Topology = Literal["open", "closed"]
InputFormat = Literal["kml", "latlon", "lonlat"]

_MAP_MAX_PASSES = 100


def haversine(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    radius: float = EARTH_RADIUS_KM,
) -> NDArray[np.float64]:
    """Great-circle distance between points given in **radians**.

    The haversine form is used rather than the spherical law of cosines because
    it stays well-conditioned for the short separations that dominate here
    (metres, against an Earth radius of thousands of kilometres).
    """
    dlat = np.subtract(lat2, lat1)
    dlon = np.subtract(lon2, lon1)
    sin_dlat = np.sin(0.5 * dlat)
    sin_dlon = np.sin(0.5 * dlon)
    a = sin_dlat * sin_dlat + np.cos(lat1) * np.cos(lat2) * (sin_dlon * sin_dlon)
    # Guard against a > 1 from round-off at antipodal-ish inputs.
    return 2.0 * radius * np.arcsin(np.sqrt(np.minimum(a, 1.0)))



def _kml_line_coordinates(text: str, source: str) -> NDArray[np.float64]:
    """(lat, lon) columns, in degrees, of the single line geometry of a KML document.

    Only ``LineString`` and ``LinearRing`` geometries are read; ``Point`` placemarks
    (markers) are ignored, since they carry no ordering along the coast. More than one
    line is refused rather than silently concatenated, because the order in which the
    lines would be joined is not defined by the file.
    """
    root = ET.fromstring(text)
    lines = [
        (elem, child.text)
        for elem in root.iter()
        if elem.tag.rpartition("}")[2] in ("LineString", "LinearRing")
        for child in elem.iter()
        if child.tag.rpartition("}")[2] == "coordinates" and child.text
    ]
    if not lines:
        raise ValueError(
            f"{source} has no LineString/LinearRing coordinates; "
            "draw the coastline as a single line before exporting"
        )
    if len(lines) > 1:
        raise ValueError(
            f"{source} has {len(lines)} line geometries; keep a single line in the file"
        )

    # KML separates tuples by whitespace and the components of a tuple by commas
    # (lon,lat[,alt]). Splitting on tuples first makes the 2-D/3-D distinction
    # unambiguous; guessing it from the total number of values is not (a 2-D file
    # whose length is a multiple of 3 would be misread, silently, wherever |lon| <= 90).
    # Whitespace around commas is not valid KML but some writers emit it.
    text = re.sub(r"\s*,\s*", ",", lines[0][1].strip())
    lon_list, lat_list = [], []
    for k, tup in enumerate(text.split(), 1):
        parts = tup.split(",")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"{source}: coordinate tuple #{k} {tup!r} has {len(parts)} "
                "components; expected lon,lat[,alt]"
            )
        try:
            lon_list.append(float(parts[0]))
            lat_list.append(float(parts[1]))
        except ValueError:
            raise ValueError(f"{source}: coordinate tuple #{k} {tup!r} is not numeric") from None
    return np.column_stack([np.asarray(lat_list), np.asarray(lon_list)])


def read_coordinates(
    path: str | Path, input_format: InputFormat
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read (lat, lon) in degrees. The format is explicit; the file extension is not used.

    ``kml``
        A KML document, or a KMZ archive (detected from its content) holding exactly
        one ``.kml`` entry. The document must contain a single LineString/LinearRing.
    ``latlon`` / ``lonlat``
        Plain text with two numeric columns in the stated order, separated by
        whitespace (spaces or tabs) or commas. Lines starting with ``#`` are comments.
        Semicolon separators and decimal commas are not supported.
    """
    path = Path(path)
    if input_format == "kml":
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if len(names) != 1:
                    raise ValueError(
                        f"{path}: expected exactly one .kml entry in the archive, "
                        f"found {len(names)}"
                    )
                text = z.read(names[0]).decode("utf-8")
            latlon = _kml_line_coordinates(text, f"{path}:{names[0]}")
        else:
            latlon = _kml_line_coordinates(path.read_text(encoding="utf-8"), str(path))
        lat, lon = latlon[:, 0], latlon[:, 1]
    elif input_format in ("latlon", "lonlat"):
        raw = path.read_text(encoding="utf-8").replace(",", " ")
        try:
            cols = np.loadtxt(raw.splitlines(), ndmin=2)
        except ValueError as exc:
            raise ValueError(
                f"{path}: not a two-column numeric table ({exc}); use spaces, tabs or "
                "commas as separators, '.' as decimal mark, and '#' for header lines"
            ) from None
        if cols.shape[1] != 2:
            raise ValueError(
                f"{path}: expected 2 numeric columns, found {cols.shape[1]} "
                "(';' separators and decimal commas are not supported)"
            )
        if input_format == "latlon":
            lat, lon = cols[:, 0], cols[:, 1]
        else:
            lat, lon = cols[:, 1], cols[:, 0]
    else:
        raise ValueError(
            f"unknown input_format {input_format!r}; expected kml|latlon|lonlat"
        )
    if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
        raise ValueError(f"{path}: non-finite coordinates")
    if np.any(np.abs(lat) > 90.0) or np.any(np.abs(lon) > 180.0):
        raise ValueError(
            f"{path}: coordinates out of range for input_format={input_format!r} "
            "(degrees expected: |lat| <= 90, |lon| <= 180)"
        )
    if lat.size < 3:
        raise ValueError(f"{path} has fewer than 3 points")
    return np.ascontiguousarray(lat), np.ascontiguousarray(lon)


def _build_interpolator(t: NDArray[np.float64], y: NDArray[np.float64], kind: InterpKind):
    """Return an interpolant exposing both ``__call__`` and ``.derivative()``.

    ``interp1d`` is deliberately avoided: it has no analytic derivative, which
    would force a finite-difference fallback in the arc-length integrator.
    ``make_interp_spline(k=1)`` gives the same piecewise-linear curve with a
    usable derivative.
    """
    if kind == "cubic":
        return CubicSpline(t, y, bc_type="not-a-knot")
    if kind == "akima":
        return Akima1DInterpolator(t, y)
    if kind == "linear":
        return make_interp_spline(t, y, k=1)
    raise ValueError(f"unknown interpolation kind {kind!r}; expected cubic|akima|linear")


class CoastlineModel:
    """Continuous coastline model with an arc-length parameterisation.

    Parameters
    ----------
    lat_deg, lon_deg
        Digitised coordinates in decimal degrees.
    kind
        Interpolation scheme for the curve itself.
    topology
        'open' leaves both endpoints intact. 'closed' explicitly represents an
        island ring with a repeated final vertex (added if absent); currently
        requires linear interpolation so the two sides meet exactly.
    radius
        Sphere radius, km.
    quad_order
        Number of Gauss-Legendre nodes per quadrature panel.
    subdivisions
        *Initial* sub-intervals per knot interval of the ``s <-> t`` map. The grid
        is then refined adaptively, so this only sets the starting point.
    map_tol
        Target for the density distortion ``max |q - 1|`` of the map (see module
        docstring). Values below ~1e-8 hit the round-off floor and buy nothing.
    map_max_nodes
        Hard cap on the size of the map grid; refinement stops (with a warning)
        if it would be exceeded.
    """

    def __init__(
        self,
        lat_deg: ArrayLike,
        lon_deg: ArrayLike,
        kind: InterpKind = "linear",
        topology: Topology = "open",
        radius: float = EARTH_RADIUS_KM,
        quad_order: int = 10,
        subdivisions: int = 16,
        map_tol: float = 1e-6,
        map_max_nodes: int = 2_000_000,
    ) -> None:
        lat = np.radians(np.asarray(lat_deg, dtype=float)).copy()
        lon = np.radians(np.asarray(lon_deg, dtype=float)).copy()
        if lat.ndim != 1 or lat.shape != lon.shape:
            raise ValueError("lat_deg and lon_deg must be 1-D arrays of equal length")
        if topology not in ("open", "closed"):
            raise ValueError("topology must be 'open' or 'closed'")
        if topology == "closed" and kind != "linear":
            raise ValueError("a closed contour currently requires kind='linear'")
        if lat.size < (3 if topology == "closed" else 4):
            raise ValueError("not enough vertices for the chosen topology")
        if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
            raise ValueError("non-finite coordinates")
        if not map_tol > 0.0:
            raise ValueError("map_tol must be positive")

        self.topology: Topology = topology
        self.closure_added = False
        self.closure_gap_km = 0.0
        if topology == "closed":
            # A closed ring contains its start once as a vertex and a second time
            # as the endpoint. Missing closure is explicit in the metadata.
            gap = float(haversine(lat[-1], lon[-1], lat[0], lon[0], radius))
            self.closure_gap_km = gap
            if gap < 1e-9:  # physically identical: remove sub-nanometre seam drift
                lat[-1] = lat[0]
                lon[-1] = lon[0]
            else:
                edges = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:], radius)
                positive = edges[edges > 0]
                median = float(np.median(positive)) if positive.size else 0.0
                if median > 0 and gap > 10 * median:
                    warnings.warn(
                        f"closing coastline with a {gap:.4g} km segment (median input "
                        f"edge {median:.4g} km): check that the data cover a whole island",
                        RuntimeWarning, stacklevel=2,
                    )
                lat = np.append(lat, lat[0])
                lon = np.append(lon, lon[0])
                self.closure_added = True

        # Unwrap *after* appending the seam: this also works when the ring crosses
        # the antimeridian or winds around a pole.
        lon = np.unwrap(lon)
        self.kind: InterpKind = kind
        self.radius = float(radius)
        self.lat_rad = lat
        self.lon_rad = lon

        # --- Layer 1: the curve, in the chordal parameter t -------------------
        seg = np.zeros(lat.size)
        seg[1:] = haversine(lat[:-1], lon[:-1], lat[1:], lon[1:], radius)
        if np.any(seg[1:] <= 0.0):
            n_dup = int(np.count_nonzero(seg[1:] <= 0.0))
            raise ValueError(
                f"{n_dup} duplicate consecutive sample(s); "
                "deduplicate the input before fitting"
            )
        self.t_knots = np.cumsum(seg)
        self.polyline_length = float(self.t_knots[-1])

        self._lat_spl = _build_interpolator(self.t_knots, lat, kind)
        self._lon_spl = _build_interpolator(self.t_knots, lon, kind)
        self._dlat = self._lat_spl.derivative()
        self._dlon = self._lon_spl.derivative()

        # --- Layer 2: the arc-length map s <-> t ------------------------------
        self.map_tol = float(map_tol)
        self._build_arc_maps(quad_order, subdivisions, self.map_tol, int(map_max_nodes))
        self.length = float(self._s_grid[-1])
        self._arc_lipschitz: float | None = None

    # ------------------------------------------------------------------ ctors
    @classmethod
    def from_file(
        cls, path: str | Path, input_format: InputFormat, **kwargs
    ) -> "CoastlineModel":
        """Load a coastline; see :func:`read_coordinates` for ``input_format``."""
        lat, lon = read_coordinates(path, input_format)
        return cls(lat, lon, **kwargs)

    # -------------------------------------------------------------- internals
    def _speed(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """|dP/dt| in km, i.e. the spherical line element."""
        lat = self._lat_spl(t)
        dlat = self._dlat(t)
        dlon = self._dlon(t)
        c = np.cos(lat)
        return self.radius * np.sqrt(dlat * dlat + c * c * dlon * dlon)

    def _panels(self, lo, hi, nodes, weights) -> NDArray[np.float64]:
        """Arc length of the cells ``[lo, hi]``, by Gauss-Legendre.

        Every grid used here contains all the knots, so no cell straddles a knot and
        the integrand is smooth on each cell, even for the piecewise-linear
        interpolant (whose derivative is discontinuous exactly at the knots).
        """
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        return np.sum(weights * self._speed(mid[:, None] + half[:, None] * nodes), axis=1) * half

    def _one_sided_speeds(self, lo, hi):
        """Speed at the start of every cell (right limit) and at its end (left limit)."""
        v_start = self._speed(lo)
        v_end = self._speed(np.nextafter(hi, -np.inf))
        if np.any(v_start <= 0.0) or np.any(v_end <= 0.0):
            raise RuntimeError("the interpolated curve stops (zero speed); check the input")
        return v_start, v_end

    @staticmethod
    def _hermite_coeffs(x, y, m_start, m_end):
        """Coefficients of the piecewise cubic Hermite through (x, y) with per-cell end
        slopes, and the Fritsch-Carlson ratios (slope / secant) used to certify
        monotonicity cell by cell."""
        h = np.diff(x)
        sec = np.diff(y) / h
        c = np.empty((4, h.size))
        c[0] = (m_start + m_end - 2.0 * sec) / (h * h)
        c[1] = (3.0 * sec - 2.0 * m_start - m_end) / h
        c[2] = m_start
        c[3] = y[:-1]
        return c, m_start / sec, m_end / sec

    @classmethod
    def _hermite(cls, x, y, m_start, m_end) -> PPoly:
        c, _, _ = cls._hermite_coeffs(x, y, m_start, m_end)
        return PPoly(c, x, extrapolate=True)

    def _cell_distortion(self, c, width, t_lo, t_hi, frac) -> NDArray[np.float64]:
        """max |q - 1| at the fractions ``frac`` of each cell of the map t(s).

        ``c`` holds the cells' Hermite coefficients and ``width`` their widths in
        ``s``. Each cell is evaluated from its own coefficients, without searching
        the breakpoints.
        """
        u = width[:, None] * frac
        c0, c1, c2, c3 = (ci[:, None] for ci in c)
        t = ((c0 * u + c1) * u + c2) * u + c3
        dt_ds = (3.0 * c0 * u + 2.0 * c1) * u + c2
        q = self._speed(np.clip(t, t_lo, t_hi)) * dt_ds
        return np.max(np.abs(q - 1.0), axis=1)

    def _build_arc_maps(
        self, quad_order: int, subdivisions: int, tol: float, max_nodes: int
    ) -> None:
        """Adaptive construction of the maps t(s) and s(t).

        Each pass: accumulate s on the current t-grid, build t(s), measure the density
        distortion |q - 1| at five interior points of every *new* cell, and split (in
        t) the cells that fail. A cell also fails if its Hermite piece is not
        certifiably monotone. The acceptance threshold of a cell is never below its
        round-off floor (~16 eps L / cell width), so refinement cannot chase noise.

        Only cells created by the last split are integrated and tested: a cell that
        is not split keeps its quadrature panel, its end slopes and its width in s,
        hence its distortion. The cheap per-cell quantities (Hermite coefficients,
        monotonicity ratios, round-off floor) are recomputed for every cell.
        """
        k = self.t_knots
        offsets = np.linspace(0.0, 1.0, max(int(subdivisions), 1) + 1)[:-1]
        grid = np.unique(
            np.concatenate([(k[:-1, None] + np.diff(k)[:, None] * offsets).ravel(), k[-1:]])
        )
        nodes, weights = np.polynomial.legendre.leggauss(quad_order)
        frac = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        eps = np.finfo(float).eps
        t_lo, t_hi = grid[0], grid[-1]

        panel = self._panels(grid[:-1], grid[1:], nodes, weights)
        v_start, v_end = self._one_sided_speeds(grid[:-1], grid[1:])
        err = np.empty(panel.size)
        fresh = np.ones(panel.size, dtype=bool)
        converged = False

        for self.map_passes in range(1, _MAP_MAX_PASSES + 1):
            s = np.concatenate(([0.0], np.cumsum(panel)))
            width = np.diff(s)
            if not np.all(width > 0.0):
                raise RuntimeError("non-monotone arc length; check for degenerate input")
            c, a, b = self._hermite_coeffs(s, grid, 1.0 / v_start, 1.0 / v_end)
            if fresh.any():
                err[fresh] = self._cell_distortion(c[:, fresh], width[fresh], t_lo, t_hi, frac)
            floor = 16.0 * eps * s[-1] / width
            not_monotone = (a <= 0) | (b <= 0) | (a * a + b * b > 9.0)
            bad = (err > np.maximum(tol, floor)) | not_monotone

            lo, hi = grid[:-1], grid[1:]
            mid = 0.5 * (lo + hi)
            split = bad & (mid > lo) & (mid < hi)
            n_split = int(np.count_nonzero(split))
            if n_split == 0:
                converged = True
                break
            # Stop *before* refining, so that grid, s, coefficients and err stay consistent.
            if grid.size + n_split > max_nodes or self.map_passes == _MAP_MAX_PASSES:
                break

            # Split: cell j becomes cells first[j] (and first[j] + 1 if split).
            counts = 1 + split.astype(np.int64)
            first = np.cumsum(counts) - counts
            n_cells = lo.size + n_split
            new_grid = np.empty(n_cells + 1)
            new_grid[first] = lo
            new_grid[first[split] + 1] = mid[split]
            new_grid[-1] = grid[-1]
            kept_old = ~split
            kept_new = first[kept_old]
            fresh = np.ones(n_cells, dtype=bool)
            fresh[kept_new] = False

            def carry(old: NDArray[np.float64]) -> NDArray[np.float64]:
                out = np.empty(n_cells)
                out[kept_new] = old[kept_old]
                return out

            panel, v_start, v_end, err = carry(panel), carry(v_start), carry(v_end), carry(err)
            grid = new_grid
            f_lo, f_hi = grid[:-1][fresh], grid[1:][fresh]
            panel[fresh] = self._panels(f_lo, f_hi, nodes, weights)
            v_start[fresh], v_end[fresh] = self._one_sided_speeds(f_lo, f_hi)

        unresolved = (err > tol) | not_monotone
        resolvable = floor <= tol
        if not converged:
            warnings.warn(
                f"s<->t map not refined to map_tol={tol:g}: reached {grid.size} nodes "
                f"after {self.map_passes} passes (max distortion {err.max():.2e}); "
                "raise map_max_nodes or map_tol",
                RuntimeWarning,
                stacklevel=3,
            )
        self._t_grid, self._s_grid = grid, s
        self._t_of_s = PPoly(c, s, extrapolate=True)
        self._s_of_t = self._hermite(grid, s, v_start, v_end)
        self.map_nodes = int(grid.size)
        self.map_converged = converged
        # Max |q - 1| over all cells, including those accepted only by the round-off
        # floor (tiny cells near points where the curve almost stops) ...
        self.map_distortion = float(err.max())
        # ... over the cells where map_tol is attainable ...
        self.map_distortion_resolvable = (
            float(err[resolvable].max()) if resolvable.any() else 0.0
        )
        # ... and the arc length (km) of the cells left above map_tol.
        self.map_unresolved_km = float(width[unresolved].sum())

    # ----------------------------------------------------------------- public
    @property
    def arc_lipschitz(self) -> float:
        """Upper estimate of |dP/ds| (km per km of *nominal* arc length ``s``).

        It would be exactly 1 if the tabulated map ``t(s)`` were exact. Interpolation
        error in that map makes the curve run locally faster or slower than unit speed
        in ``s``; this number is the worst case. Any chord measured from a fixed station
        therefore grows by at most ``arc_lipschitz * ds`` when the other end moves by
        ``ds``, which is what lets the yardstick search take provably safe steps.

        Evaluated once, lazily, on 16 interior points of every cell of the map. Map
        cells never straddle a knot, so the quantity is smooth on each cell.
        """
        if self._arc_lipschitz is None:
            s = self._s_grid
            frac = (np.arange(16) + 0.5) / 16.0
            ss = np.concatenate([(s[:-1, None] + np.diff(s)[:, None] * frac).ravel(), s])
            dt_ds = self._t_of_s.derivative()(ss)
            q = self._speed(self.t_of_s(ss)) * dt_ds
            self._arc_lipschitz = float(max(np.max(q), 1.0))
        return self._arc_lipschitz

    def t_of_s(self, s: ArrayLike) -> NDArray[np.float64]:
        """Chordal parameter at arc length ``s`` (clipped to the domain)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        # Clip the result too: the last ulp of the interpolant can step outside the
        # domain, and Akima returns NaN there instead of extrapolating.
        return np.clip(self._t_of_s(s), self._t_grid[0], self._t_grid[-1])

    def s_of_t(self, t: ArrayLike) -> NDArray[np.float64]:
        """Arc length at chordal parameter ``t`` (clipped to the domain)."""
        t = np.clip(np.asarray(t, dtype=float), self._t_grid[0], self._t_grid[-1])
        return np.clip(self._s_of_t(t), 0.0, self.length)

    def points(self, s: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(lat, lon) in radians at arc length ``s``. Shape-preserving."""
        t = self.t_of_s(s)
        return self._lat_spl(t), self._lon_spl(t)

    @property
    def t_span(self) -> Tuple[float, float]:
        """Domain of the chordal parameter, for runs that deliberately sample in ``t``."""
        return float(self._t_grid[0]), float(self._t_grid[-1])

    def points_of_t(self, t: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(lat, lon) in radians at chordal parameter ``t``."""
        t = np.clip(np.asarray(t, dtype=float), self._t_grid[0], self._t_grid[-1])
        return self._lat_spl(t), self._lon_spl(t)

    def chord(self, s_a: ArrayLike, s_b: ArrayLike) -> NDArray[np.float64]:
        """Straight-line (great-circle) distance between two arc-length stations."""
        lat_a, lon_a = self.points(s_a)
        lat_b, lon_b = self.points(s_b)
        return haversine(lat_a, lon_a, lat_b, lon_b, self.radius)

    def polyline_length_of(self, s: NDArray[np.float64], axis: int = -1) -> NDArray[np.float64]:
        """Summed chord lengths of the polyline through stations ``s``."""
        lat, lon = self.points(s)
        sl_lo = [slice(None)] * lat.ndim
        sl_hi = [slice(None)] * lat.ndim
        sl_lo[axis] = slice(None, -1)
        sl_hi[axis] = slice(1, None)
        d = haversine(
            lat[tuple(sl_lo)], lon[tuple(sl_lo)],
            lat[tuple(sl_hi)], lon[tuple(sl_hi)],
            self.radius,
        )
        return d.sum(axis=axis)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CoastlineModel(topology={self.topology!r}, kind={self.kind!r}, n_samples={self.lat_rad.size}, "
            f"length={self.length:.6f} km, polyline={self.polyline_length:.6f} km, "
            f"map_nodes={self.map_nodes}, map_distortion={self.map_distortion:.1e})"
        )
