"""Local Richardson dimension versus scale, for Buzios and the two test curves.

Usage:  python calibration/plot_local_dimension.py RESULTS_DIR OUTPUT.png

Reads ``<RESULTS_DIR>/<name>_RP/<name>_RP_pdfs.npz`` for name in buzios, koch, sine
(written by calibration/run_calibration.sh) and plots the stored
``richardson_dimension`` against ``effective_ruler_km``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CURVES = [  # name, label, colour, marker
    ("buzios", "Búzios Peninsula", "#2a78d6", "o"),
    ("koch", "Koch curve (D = 1.262)", "#eb6834", "s"),
    ("sine", "Smooth sinusoid (D = 1)", "#1baf7a", "^"),
]
D_KOCH = np.log(4.0) / np.log(3.0)


def main() -> None:
    results, out = Path(sys.argv[1]), Path(sys.argv[2])
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # Buzios fit window (n_p = 10..50 maps to these effective rulers)
    z = np.load(results / "buzios_RP" / "buzios_RP_pdfs.npz")
    n, e = z["resolution"], z["effective_ruler_km"]
    lo, hi = np.interp([50, 10], n, e)
    ax.axvspan(lo, hi, color="#2a78d6", alpha=0.08, lw=0)
    ax.text(lo * 1.04, 1.785, "Búzios\nfit window\n$n_p$ = 10–50",
            ha="left", va="top", fontsize=8.5, color=INK_2)

    for d, txt in ((1.0, "D = 1"), (D_KOCH, "Koch D = 1.262")):
        ax.axhline(d, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(0.0105, d + 0.012, txt, fontsize=8, color=INK_2, va="bottom")

    for name, label, colour, marker in CURVES:
        z = np.load(results / f"{name}_RP" / f"{name}_RP_pdfs.npz")
        e, d = z["effective_ruler_km"], z["richardson_dimension"]
        order = np.argsort(e)
        e, d = e[order], d[order]
        keep = np.isfinite(d) & (d < 1.8)
        ax.plot(e[keep], d[keep], color=colour, lw=2, marker=marker, ms=4.5,
                markevery=3, label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(0.01, 10)
    ax.set_ylim(0.95, 1.8)
    ax.set_xlabel(r"Effective ruler $\langle L\rangle\,/\,(n_p - 1)$  [km]", color=INK, fontsize=10)
    ax.set_ylabel(r"Local dimension  $1 - d\ln\langle L\rangle / d\ln(\mathrm{ruler})$", color=INK, fontsize=10)
    ax.set_title("Random points: local dimension versus scale", color=INK,
                 fontsize=11.5, loc="left")
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
