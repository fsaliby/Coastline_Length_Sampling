#!/usr/bin/env bash
set -euo pipefail

python3 coastline_run.py \
--input-file koch_8_theta60.txt \
--input-format latlon \
--method rp \
--rp-sweep nested \
--save-samples \
--rp-start 5 \
--rp-stop 1000 \
--rp-step 1 \
--interpolation linear \
--quad-order 10 \
--subdivisions 1 \
--map-tol 1e-6 \
--uniform-in arc \
--engine numba \
--n-jobs -1 \
--chunk-elements 1048576 \
--n-realisations 100000 \
--n-bins 80 \
--seed 20261110 \
--output-dir koch_8_results \
--prefix koch_8_RP 

python3 coastline_mc_fit.py \
--input koch_8_results/koch_8_RP/koch_8_RP_pdfs.npz \
--n-sim 100000 \
--fit nonlinear \
--weights none \
--seed 20261110 \
--np-min 100 \
--np-max 1000 \
--curves realisation \
--nonlinear-solver profile \
--beta-xtol 1.0e-8 \
--output-bins 80 \
--joint-bins 80 \
--hist-tail 0 \
--chunk-elements 1048576 \
--output-dir koch_8_results \
--prefix koch_8_RP
