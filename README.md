CMAPSS Turbo Fan Engine RUL Prediction 
======================================

What this folder contains
-------------------------
This package documents Remaining Useful Life (RUL) prediction
on NASA's CMAPSS turbofan datasets FD001 and FD002.

- **README.md** -          this file (report + Python code overview)
- **CMAPPS.pdf** -         PDF report 
- **Python** -         Python codes used for producing the results. CMAPPSturboengine.py and CMAPPSturboengine2.py are what produces the plots in CMAPPS.pdf , whereas CMAPPSturboengine3saveModels.py and CMAPPSturboengine4FastAPI create and use the trained prediction models for faster use. The fast API is also partly available as a docker image.

### Docker Image

**Image name:** `joeysmit9717/cmapps-turboengine`

**Docker Hub:** [joeysmit9717/cmapps-turboengine]
(https://hub.docker.com/r/joeysmit9717/cmapps-turboengine)

To download the image, run:
docker pull joeysmit9717/cmapps-turboengine


  Upstream analysis scripts and data:
  CMAPPSturboengine.py   - FD001 (1 operating condition, 1 fault mode)
  CMAPPSturboengine2.py  - FD002 (6 operating conditions, 1 fault mode)
  train_FD001.txt / test_FD001.txt / RUL_FD001.txt
  train_FD002.txt / test_FD002.txt / RUL_FD002.txt

- **train:** full life of the engine, engine inoperational at the end
- **test:** data cut-off after X cycles, engine still operational at that pint
- **RUL:** addition to test, shows how many more cycles the engine lasted until it became inoperational.

Key Findings
------------
![FD001 Training Trajectories](fd001_trajectory_train.png)

What the analysis does
----------------------
1. Load CMAPSS run-to-failure training trajectories and truncated test
   engines; attach linear RUL labels on train (max cycle - current cycle).
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

Plots in CMAPPSturboengine.py and CMAPPSturboengine2.py
-----------------
 - fd001_parity.png                 Predicted vs true RUL (FD001)
 - fd001_ecdf_relative_error.png    Relative-error ECDF (FD001)
 - fd001_trajectory_test.png        Test engine 1 RUL trajectory
 - fd001_trajectory_train.png       Train engine (run-to-failure)
 - fd002_parity.png                 Predicted vs true RUL (FD002)
 - fd002_ecdf_relative_error.png    Relative-error ECDF (FD002)
 - fd002_trajectory_test.png        Test engine 1 RUL trajectory
 - fd002_trajectory_train.png       Train engine (run-to-failure)

How to regenerate the plots
---------------------------
From the project root (or any cwd; paths are resolved from the scripts):

 - set MPLBACKEND=Agg
 - CMAPPSturboengine.py
 - CMAPPSturboengine2.py

Requirements
------------
Python 3.x with: pandas, numpy, matplotlib, scikit-learn, tensorflow, NASA CMAPPS dataset FD001 and FD002 (train, test and RUL)

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
