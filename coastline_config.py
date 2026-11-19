"""Run configuration: a single validated, serialisable object."""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

from coastline_geometry import InputFormat, InterpKind

__all__ = ["RunConfig", "build_parser", "config_from_args"]

Method = Literal["rp", "ys"]
Topology = Literal["open", "closed"]
EndPolicy = Literal["merge", "keep", "drop"]
UniformIn = Literal["arc", "chordal"]
Engine = Literal["numpy", "numba"]
RpSweep = Literal["independent", "nested"]
RpSpacing = Literal["linear", "log"]
YsSeeds = Literal["independent", "common"]
YsWalker = Literal["auto", "march", "polyline"]


_INPUT_FORMATS = ("kml", "latlon", "lonlat")
_INTERP_KINDS = ("cubic", "akima", "linear")
_END_POLICIES = ("merge", "keep", "drop")
_REQUIRED = ("input_file", "input_format")


@dataclass
class RunConfig:
    input_file: str
    input_format: InputFormat
    method: Method = "rp"
    topology: Topology = "open"

    rp_start: int = 5
    rp_stop: int = 1000
    rp_step: int = 1
    rp_spacing: RpSpacing = "linear"
    rp_per_decade: int = 20
    ys_start: float = 0.05
    ys_stop: float = 10.0
    ys_num: int = 10

    n_realisations: int = 10_000
    n_bins: int = 50
    seed: int = 20250101

    interpolation: InterpKind = "linear"
    quad_order: int = 10
    subdivisions: int = 16
    map_tol: float = 1e-6
    map_max_nodes: int = 2_000_000

    end_policy: EndPolicy = "merge"
    uniform_in: UniformIn = "arc"
    rp_sweep: RpSweep = "nested"
    ys_seeds: YsSeeds = "common"
    ys_walker: YsWalker = "auto"

    engine: Engine = "numpy"
    n_jobs: int = 1
    chunk_elements: int = 1_048_576

    output_dir: str = "results"
    prefix: str = "run"
    save_samples: bool = False

    def __post_init__(self) -> None:
        if self.input_format not in _INPUT_FORMATS:
            raise ValueError(f"input_format must be one of {_INPUT_FORMATS}")
        if self.interpolation not in _INTERP_KINDS:
            raise ValueError(f"interpolation must be one of {_INTERP_KINDS}")
        if self.end_policy not in _END_POLICIES:
            raise ValueError(f"end_policy must be one of {_END_POLICIES}")
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be >= 1, or negative (joblib convention: -1 = all cores)")
        if self.topology not in ("open", "closed"):
            raise ValueError("topology must be 'open' or 'closed'")
        if self.topology == "closed" and self.interpolation != "linear":
            raise ValueError("topology='closed' currently requires interpolation='linear' "
                             "to preserve the polygonal closure without a spline seam")
        if self.method not in ("rp", "ys"):
            raise ValueError("method must be 'rp' or 'ys'")
        if self.n_realisations < 2:
            raise ValueError("n_realisations must be >= 2 for a variance to exist")
        if self.n_bins < 10:
            raise ValueError("n_bins must be >= 10")
        if self.method == "rp":
            if self.rp_start < 3:
                raise ValueError("rp_start must be >= 3")
            if self.rp_spacing not in ("linear", "log"):
                raise ValueError("rp_spacing must be 'linear' or 'log'")
            if self.rp_spacing == "linear":
                if self.rp_step < 1 or self.rp_stop < self.rp_start:
                    raise ValueError("require rp_step >= 1 and rp_stop >= rp_start")
            else:
                if self.rp_stop <= self.rp_start:
                    raise ValueError("log spacing requires rp_stop > rp_start")
                if self.rp_per_decade < 1:
                    raise ValueError("rp_per_decade must be >= 1")
        else:
            if not (0.0 < self.ys_start < self.ys_stop):
                raise ValueError("require 0 < ys_start < ys_stop")
            if self.ys_num < 2:
                raise ValueError("ys_num must be >= 2")
        if self.uniform_in not in ("arc", "chordal"):
            raise ValueError("uniform_in must be 'arc' or 'chordal'")
        if self.rp_sweep not in ("independent", "nested"):
            raise ValueError("rp_sweep must be 'independent' or 'nested'")
        if self.ys_seeds not in ("independent", "common"):
            raise ValueError("ys_seeds must be 'independent' or 'common'")
        if self.ys_walker not in ("auto", "march", "polyline"):
            raise ValueError("ys_walker must be 'auto', 'march', or 'polyline'")
        # Resolve the conditional default before validating compatibility. An explicit
        # choice of 'polyline' with a non-linear interpolant must still fail.
        # Store the resolved walker so runs and archives never contain an ambiguous 'auto'.
        if self.ys_walker == "auto":
            self.ys_walker = "polyline" if self.interpolation == "linear" else "march"
        # Like the other YS options, ys_walker is ignored by method 'rp'. The upper
        # limit on the ruler depends on the data and is checked by the campaign driver.
        if self.method == "ys" and self.ys_walker == "polyline" and self.interpolation != "linear":
            raise ValueError(
                "--ys-walker polyline requires --interpolation linear.\n"
                f"  Received: --ys-walker polyline, --interpolation {self.interpolation}.\n"
                "  Valid alternatives: --ys-walker march with the selected interpolation, "
                "or --interpolation linear with --ys-walker polyline."
            )
        if self.topology == "closed" and self.method == "ys":
            if self.ys_walker != "polyline":
                raise ValueError("closed YS requires ys_walker='polyline' for exact periodic intersections")
            if self.end_policy != "keep":
                raise ValueError("closed YS requires end_policy='keep' (one closing chord)")
        if self.engine not in ("numpy", "numba"):
            raise ValueError("engine must be 'numpy' or 'numba'")
        if self.engine == "numba" and importlib.util.find_spec("numba") is None:
            raise ValueError("engine='numba' needs the numba package (pip install numba)")
        if not self.map_tol > 0.0:
            raise ValueError("map_tol must be positive")
        if self.quad_order < 2 or self.subdivisions < 1 or self.map_max_nodes < 10:
            raise ValueError("require quad_order >= 2, subdivisions >= 1, map_max_nodes >= 10")
        if self.chunk_elements < 1:
            raise ValueError("chunk_elements must be >= 1")

    @property
    def resolutions(self) -> NDArray[np.float64] | NDArray[np.int64]:
        if self.method == "rp":
            if self.rp_spacing == "linear":
                # rp_stop is inclusive but never exceeded, whatever rp_step.
                return np.arange(self.rp_start, self.rp_stop + 1, self.rp_step, dtype=np.int64)

            n_intervals = max(
                1,
                int(np.rint(
                    self.rp_per_decade * np.log10(self.rp_stop / self.rp_start)
                )),
            )
            return np.unique(
                np.rint(
                    np.geomspace(self.rp_start, self.rp_stop, n_intervals + 1)
                ).astype(np.int64)
            )

        return np.geomspace(self.ys_start, self.ys_stop, self.ys_num, dtype=float)

    @property
    def resolution_label(self) -> str:
        return "n_points" if self.method == "rp" else "ruler_km"

    @property
    def out_dir(self) -> Path:
        return Path(self.output_dir) / self.prefix

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def read_json(path: str | Path) -> Dict[str, Any]:
        """Raw keys of a JSON config, checked against the fields but not validated,
        so that command-line flags can complete or override them first."""
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: a JSON object is expected")
        unknown = set(raw) - {f.name for f in fields(RunConfig)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return raw

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        """A complete, validated configuration from a JSON file."""
        return cls(**cls.read_json(path))


def build_parser() -> argparse.ArgumentParser:
    """CLI parser. Every flag defaults to None so that, when absent, the value from
    ``--config`` (or else the RunConfig default) is used; the help text shows the
    RunConfig default instead of None."""
    defaults = {f.name: f.default for f in fields(RunConfig) if f.default is not MISSING}
    p = argparse.ArgumentParser(
        prog="coastline",
        description="Monte-Carlo sampling distributions of measured coastline length.",
    )

    def add(group, flag: str, help: str, **kw) -> None:
        key = flag.lstrip("-").replace("-", "_")
        if key in defaults:
            help = f"{help} [default: {defaults[key]}]"
        group.add_argument(flag, help=help, **kw)

    p.add_argument("--config", type=Path, help="JSON config file; command-line flags override it")
    add(p, "--input-file", "coastline file; see --input-format. Required, here or in "
        "--config", type=str)
    add(p, "--input-format", "kml = KML/KMZ with a single line; latlon / lonlat = text with "
        "two columns in that order (degrees; separated by spaces, tabs or commas; '#' "
        "comments). The file extension is not used. Required, here or in --config",
        choices=["kml", "latlon", "lonlat"])
    add(p, "--method", "rp = random points, ys = yardstick", choices=["rp", "ys"])
    add(p, "--topology", "open = fixed endpoints; closed = periodic island/perimeter "
        "(requires linear interpolation; closed YS: polyline walker + keep)",
        choices=["open", "closed"])

    g = p.add_argument_group("random points sweep (method rp)")
    add(g, "--rp-start", "first number of stations (>= 3; closed: includes the repeated start)", type=int)
    add(g, "--rp-stop", "last number of stations (inclusive)", type=int)
    add(g, "--rp-spacing", "spacing of n_points values", choices=["linear", "log"])
    add(g, "--rp-step", "increment of n_points when --rp-spacing=linear", type=int)
    add(g, "--rp-per-decade", "logarithmic intervals per decade when --rp-spacing=log", type=int)

    g = p.add_argument_group("yardstick sweep (method ys; km, geometric spacing)")
    add(g, "--ys-start", "smallest ruler, km", type=float)
    add(g, "--ys-stop", "largest ruler, km", type=float)
    add(g, "--ys-num", "number of rulers (>= 2)", type=int)

    g = p.add_argument_group("monte carlo")
    add(g, "--n-realisations", "realisations per resolution (>= 2)", type=int)
    add(g, "--n-bins", "histogram bins per resolution (>= 10)", type=int)
    add(g, "--seed", "master seed", type=int)

    g = p.add_argument_group("model")
    add(g, "--interpolation", "curve through the input samples",
        choices=["cubic", "akima", "linear"])
    add(g, "--quad-order", "Gauss-Legendre nodes per quadrature panel", type=int)
    add(g, "--subdivisions", "initial cells per knot interval of the s<->t map "
        "(refined adaptively afterwards)", type=int)
    add(g, "--map-tol", "max density distortion |q-1| of the s<->t map", type=float)
    add(g, "--map-max-nodes", "cap on the s<->t map grid size", type=int)

    g = p.add_argument_group("method options")
    add(g, "--ys-seeds", "YS: 'common' starts every ruler of a realisation from the same "
        "seed station (same P(L|d) for each ruler, columns become whole surveys)",
        choices=["independent", "common"])
    add(g, "--ys-walker", "YS: how each station is found. 'auto' chooses 'polyline' for "
        "linear interpolation and 'march' for akima/cubic; 'march' = Lipschitz march + "
        "root bracketing; 'polyline' = exact segment-by-segment walk "
        "(requires linear interpolation and rulers below the pole-distance limit). "
        "Explicitly incompatible choices are errors",
        choices=["auto", "march", "polyline"])
    add(g, "--end-policy", "open YS: two end fragments; closed YS: 'keep' only (closing chord)",
        choices=["merge", "keep", "drop"])
    add(g, "--uniform-in", "RP: draw stations uniformly in true arc length or in the "
        "chordal parameter", choices=["arc", "chordal"])
    add(g, "--rp-sweep", "RP: 'nested' reuses one draw per realisation for every "
        "n_points (exact P(L|n) for each n, dependent across n)",
        choices=["independent", "nested"])

    g = p.add_argument_group("execution / output")
    add(g, "--engine", "'numba' = compiled kernels (needs numba); same results up to "
        "round-off", choices=["numpy", "numba"])
    add(g, "--n-jobs", "worker processes; -1 uses every core", type=int)
    add(g, "--chunk-elements", "elements per array in vectorised blocks (memory knob; "
        "results do not depend on it)", type=int)
    add(g, "--output-dir", "parent directory of the outputs", type=str)
    add(g, "--prefix", "run name: sub-directory and file prefix", type=str)
    add(g, "--save-samples", "RP: also store every raw length sample (YS samples are "
        "always stored)", action="store_true", default=None)
    add(g, "--no-save-samples", "RP: do not store the raw samples (only their "
        "summary and histograms); the default is not to save samples",
        dest="save_samples", action="store_false", default=None)
    return p


def config_from_args(argv: Sequence[str] | None = None) -> RunConfig:
    args = build_parser().parse_args(argv)
    # The JSON file is merged *before* validation: it may be incomplete or invalid on
    # its own and only become valid once the command-line flags are applied.
    base: Dict[str, Any] = RunConfig.read_json(args.config) if args.config else {}
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    merged = {**base, **overrides}
    missing = [k for k in _REQUIRED if k not in merged]
    if missing:
        flags = ", ".join("--" + k.replace("_", "-") for k in missing)
        raise SystemExit(f"error: {flags} required (on the command line or in --config)")
    return RunConfig(**merged)
