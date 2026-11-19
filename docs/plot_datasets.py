"""Local Richardson dimension versus scale for the three example datasets.

Usage (from the repository root, after running the three RP example scripts):

    python docs/plot_datasets.py [OUTPUT.png]

Archive paths can be overridden with BUZIOS=..., KOCH=..., BRITAIN=... environment
variables. Each curve is drawn thin over its whole sweep and thick over the n_p
window used by its fit.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
D_KOCH = np.log(4.0) / np.log(3.0)

DATASETS = [  # env var, default archive, label, colour, marker, fit window (n_p)
    ("KOCH", "koch_8_results/koch_8_RP/koch_8_RP_pdfs.npz",
     "Koch curve, level 8", "#eb6834", "s", (100, 1000)),
    ("BUZIOS", "Buzios_Peninsula_kml_results/Buzios_Peninsula_kml_RP/Buzios_Peninsula_kml_RP_pdfs.npz",
     "Búzios Peninsula", "#2a78d6", "o", (10, 50)),
    ("BRITAIN", "Britain_West_results/Britain_West_RP/Britain_West_RP_pdfs.npz",
     "Britain, west coast (GSHHG)", "#1baf7a", "^", (20, 200)),
]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/local_dimension_datasets.png")
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(D_KOCH, color=INK_2, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(250, D_KOCH + 0.006, "Koch exact, D = 1.262", fontsize=8, color=INK_2,
            ha="right", va="bottom")

    for env, default, label, colour, marker, (n_lo, n_hi) in DATASETS:
        z = np.load(os.environ.get(env, default))
        n, e, d = z["resolution"], z["effective_ruler_km"], z["richardson_dimension"]
        ok = np.isfinite(d)
        n, e, d = n[ok], e[ok], d[ok]
        ax.plot(e, d, color=colour, lw=1.0, alpha=0.6, zorder=2)
        win = (n >= n_lo) & (n <= n_hi)
        ax.plot(e[win], d[win], color=colour, lw=3.0, marker=marker, ms=5,
                markevery=max(1, win.sum() // 6), zorder=3,
                label=f"{label}  (fit: $n_p$ = {n_lo}–{n_hi})")

    ax.set_xscale("log")
    ax.set_xlim(0.02, 300)
    ax.set_ylim(1.0, 1.45)
    ax.set_xlabel(r"Effective ruler $\langle L\rangle\,/\,(n_p - 1)$  [km]", color=INK, fontsize=10)
    ax.set_ylabel(r"Local dimension  $1 - d\ln\langle L\rangle / d\ln(\mathrm{ruler})$",
                  color=INK, fontsize=10)
    ax.set_title("Random points: local dimension of the example datasets "
                 "(thick: fit window)", color=INK, fontsize=11, loc="left")
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
