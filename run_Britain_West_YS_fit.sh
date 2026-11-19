#!/usr/bin/env bash
set -euo pipefail

python3 coastline_run.py \
--input-file britain_west_gshhg.txt \
--input-format latlon \
--method ys \
--ys-start  13.71 \
--ys-stop 74.756 \
--ys-num 20 \
--ys-seeds common \
--ys-walker polyline \
--end-policy merge \
--n-realisations 1000000 \
--n-bins 80 \
--seed 20261110 \
--interpolation linear \
--quad-order 10 \
--subdivisions 1 \
--engine numba \
--n-jobs -1 \
--map-tol 1e-6 \
--map-max-nodes 2000000 \
--chunk-elements 1048576 \
--output-dir Britain_West_results \
--prefix Britain_West_YS_fit 

python3 coastline_mc_fit.py \
--input Britain_West_results/Britain_West_YS_fit/Britain_West_YS_fit_pdfs.npz \
--n-sim 1000000 \
--fit nonlinear \
--weights none \
--seed 20261110 \
--np-min 5 \
--np-max 300 \
--curves realisation \
--nonlinear-solver profile \
--beta-xtol 1.0e-8 \
--output-bins 80 \
--joint-bins 80 \
--hist-tail 0 \
--chunk-elements 1048576 \
--output-dir Britain_West_results \
--prefix Britain_West_YS_fit

