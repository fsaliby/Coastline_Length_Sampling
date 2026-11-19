#!/usr/bin/env bash
set -euo pipefail

python3 coastline_run.py \
--input-file koch_8_theta60.txt \
--input-format latlon \
--method ys \
--ys-start 0.063 \
--ys-stop 0.439 \
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
--output-dir koch_8_results \
--prefix koch_8_YS_fit 

python3 coastline_mc_fit.py \
--input koch_8_results/koch_8_YS_fit/koch_8_YS_fit_pdfs.npz \
--n-sim 1000000 \
--fit nonlinear \
--weights none \
--seed 20261110 \
--np-min 10 \
--np-max 1100 \
--curves realisation \
--nonlinear-solver profile \
--beta-xtol 1.0e-8 \
--output-bins 80 \
--joint-bins 80 \
--hist-tail 0 \
--chunk-elements 1048576 \
--output-dir koch_8_results \
--prefix koch_8_YS_fit

