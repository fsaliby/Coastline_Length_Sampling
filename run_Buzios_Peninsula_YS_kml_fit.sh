#!/usr/bin/env bash
set -euo pipefail

python3 coastline_run.py \
--input-file Buzios_Peninsula.kml \
--input-format kml \
--topology open \
--method ys \
--ys-start 0.585 \
--ys-stop 2.34 \
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
--output-dir Buzios_Peninsula_kml_results \
--prefix Buzios_Peninsula_kml_fit_YS 

python3 coastline_mc_fit.py \
--input Buzios_Peninsula_kml_results/Buzios_Peninsula_kml_fit_YS/Buzios_Peninsula_kml_fit_YS_pdfs.npz \
--n-sim 1000000 \
--fit nonlinear \
--weights none \
--seed 20261110 \
--np-min 5 \
--np-max 100 \
--curves realisation \
--nonlinear-solver profile \
--beta-xtol 1.0e-8 \
--output-bins 80 \
--joint-bins 80 \
--hist-tail 0 \
--chunk-elements 1048576 \
--output-dir Buzios_Peninsula_kml_results \
--prefix Buzios_Peninsula_kml_fit_YS

