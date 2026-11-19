#!/usr/bin/env bash
# Calibration: run the Buzios settings on curves of known dimension.
#
# Run from the repository root:   bash calibration/run_calibration.sh
# Realisation counts can be raised: N_RP=100000 N_YS=1000000 bash calibration/run_calibration.sh
#
# Produces calibration/results/ (archives, fits) and docs/local_dimension.png.
set -euo pipefail

N_RP="${N_RP:-10000}"
N_YS="${N_YS:-20000}"
OUT=calibration/results

python3 calibration/make_test_curves.py calibration

run_curve () {  # name, input file, input format
  local name=$1 file=$2 fmt=$3

  python3 coastline_run.py --input-file "$file" --input-format "$fmt" \
    --method rp --rp-sweep nested --save-samples \
    --rp-start 5 --rp-stop 1000 --rp-spacing log --rp-per-decade 40 \
    --interpolation linear --subdivisions 1 --engine numba --n-jobs -1 \
    --n-realisations "$N_RP" --seed 20261110 \
    --output-dir "$OUT" --prefix "${name}_RP" > /dev/null

  python3 coastline_run.py --input-file "$file" --input-format "$fmt" \
    --method ys --ys-start 0.585 --ys-stop 2.34 --ys-num 20 \
    --ys-seeds common --ys-walker polyline --end-policy merge \
    --interpolation linear --subdivisions 1 --engine numba --n-jobs -1 \
    --n-realisations "$N_YS" --seed 20261110 \
    --output-dir "$OUT" --prefix "${name}_YS" > /dev/null

  for m in RP YS; do
    if [ "$m" = RP ]; then range="--np-min 10 --np-max 50"; else range="--np-min 5 --np-max 100"; fi
    echo "== ${name} ${m}"
    python3 coastline_mc_fit.py --input "$OUT/${name}_${m}/${name}_${m}_pdfs.npz" \
      --curves realisation --fit nonlinear --weights none $range \
      --beta-xtol 1e-8 --hist-tail 0 \
      --output-dir "$OUT" --prefix "${name}_${m}" | grep -E "^  Df"
  done
}

run_curve koch   calibration/koch.txt   latlon
run_curve sine   calibration/sine.txt   latlon
run_curve buzios Buzios_Peninsula.kml   kml

python3 calibration/plot_local_dimension.py "$OUT" docs/local_dimension.png
