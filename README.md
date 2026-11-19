CMAPSS Turbo Fan Engine RUL Prediction
======================================

What this folder contains
-------------------------
This package documents Remaining Useful Life (RUL) prediction
on NASA's CMAPSS turbofan datasets FD001 and FD002.

- **README.md** -          this file (report + Python code overview)
- **CMAPPS.pdf** -          LaTeX source for the PDF findings report (Overleaf)
- **Python** -         Python codes used for producing the results. 

  Upstream analysis scripts and data (PdM/):
  CMAPPSturboengine.py   - FD001 (1 operating condition, 1 fault mode)
  CMAPPSturboengine2.py  - FD002 (6 operating conditions, 1 fault mode)
  train_FD001.txt / test_FD001.txt / RUL_FD001.txt
  train_FD002.txt / test_FD002.txt / RUL_FD002.txt

What the analysis does
----------------------
1. Load CMAPSS run-to-failure training trajectories and truncated test
   engines; attach linear RUL labels on train (max cycle - cycle).
2. Use 10 trending sensors (2, 3, 4, 7, 11, 12, 15, 17, 20, 21).
3. FD001: MinMax-scale sensors for the CNN; GPR/RF use raw sensor rows.
4. FD002: cluster op_settings into 6 regimes (KMeans), then z-score
   sensors within each regime so degradation is comparable across
   flight conditions (fit on train only).
5. Train four model families per dataset:
   - Gaussian Process Regressor on a random row sample
   - GPR on the first N engines (full trajectories)
   - Random Forest on all training rows
   - 1D CNN on sliding windows of 30 cycles
6. Evaluate on the official test set: predict RUL at each engine's last
   recorded cycle and compare to RUL_FD00x.
7. Report RMSE, MAE, and PBF (% of engines where predicted RUL < true
   RUL — more conservative / safer for scheduling).
8. Plot parity charts, ECDFs of relative lifespan error, and RUL
   trajectories on one test and one train engine.

Plots in figures/
-----------------
  fd001_parity.png                 Predicted vs true RUL (FD001)
  fd001_ecdf_relative_error.png    Relative-error ECDF (FD001)
  fd001_trajectory_test.png        Test engine 1 RUL trajectory
  fd001_trajectory_train.png       Train engine (run-to-failure)
  fd002_parity.png                 Predicted vs true RUL (FD002)
  fd002_ecdf_relative_error.png    Relative-error ECDF (FD002)
  fd002_trajectory_test.png        Test engine 1 RUL trajectory
  fd002_trajectory_train.png       Train engine (run-to-failure)

How to regenerate the plots
---------------------------
From the project root (or any cwd; paths are resolved from the scripts):

  set MPLBACKEND=Agg
  py -3 PdM\CMAPPSturboengine.py
  py -3 PdM\CMAPPSturboengine2.py

On PowerShell:

  $env:MPLBACKEND='Agg'
  py -3 PdM\CMAPPSturboengine.py
  py -3 PdM\CMAPPSturboengine2.py

Figures are written to this folder under figures/. Set SAVE_FIGURES=0 to
skip saving PNGs. FD002's 7000-row GPR is slow (several minutes).

How to build the PDF report (Overleaf)
--------------------------------------
1. Create a new Overleaf project.
2. Upload report.tex and the figures/ folder (keep the same relative
   paths: figures/*.png).
3. Compile. Download the PDF for your portfolio / applications.

Requirements
------------
Python 3.x with: pandas, numpy, matplotlib, scikit-learn, tensorflow

Key result snapshot (from last run)
-----------------------------------
  FD001 (100 test engines):
    Best RMSE: 1D CNN 26.0 | GPR-15eng 28.4 | GPR-3k 32.1 | RF 33.8
    Best MAE:  1D CNN 18.7
    Most conservative PBF: GPR-15eng 48%

  FD002 (259 test engines, regime-normalized):
    Best RMSE: 1D CNN 31.1 | GPR-20eng 32.8 | GPR-7k 34.2 | RF 36.7
    Best MAE:  1D CNN 22.5
    Most conservative PBF: GPR-20eng 54%

  Takeaway: the windowed CNN leads on absolute error on both subsets;
  trajectory GPRs are more conservative (higher PBF). Multi-regime FD002
  is harder than single-regime FD001 even after regime normalization.
