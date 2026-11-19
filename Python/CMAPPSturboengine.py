import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

#used for dataset 1  (1 operating conditions, 1 Fault mode)

tf.random.set_seed(42)   # reproducibility, mirroring random_state=42 used elsewhere

# Portfolio export folder (README / LaTeX / PNGs live here).
PORTFOLIO_DIR = Path(__file__).resolve().parent.parent / "portfolio" / "CMAPPSturbo"
FIGURES_DIR = PORTFOLIO_DIR / "figures"
# Set SAVE_FIGURES=0 to skip writing PNGs (interactive exploration only).
SAVE_FIGURES = os.environ.get("SAVE_FIGURES", "1") != "0"


def save_figure(filename):
    """Save the current matplotlib figure into the portfolio figures folder."""
    if not SAVE_FIGURES:
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def maybe_show():
    """Show the current figure unless running headless (MPLBACKEND=Agg)."""
    if os.environ.get("MPLBACKEND", "").lower() != "agg":
        plt.show()



#make names for the columns
cols = ["engine_id", "cycle"] + [f"op_setting_{i}" for i in range(1, 4)] \
       + [f"sensor_{i}" for i in range(1, 22)]

#robust way to import data files (even if they are not in the Current Working Directory)
file_path = Path(__file__).resolve().parent / "train_FD001.txt"
df = pd.read_csv(file_path, sep='\s+', header=None, names=cols)
# Add RUL number to each row based on the max cycle for each engine (grouped by engine id)
df['RUL'] = df.groupby('engine_id')['cycle'].transform('max') - df['cycle']

#some basic data about df
print('head')
print(df.head())
print('tail')
print(df.tail())
print(df.groupby('engine_id')['cycle'].max())    #max cycle before failure for each engine

### GPR with 3000 random rows
#select the data from the df for predicting 
sample = df.sample(n=3000, random_state=42)   # 3000x3000 matrix ≈ 72 MB, fits & fits fast
X_train = sample[[f"sensor_{i}" for i in [2,3,4,7,11,12,15,17,20,21]]].values
y_train = sample['RUL'].values

#settings for gpr
kernel = ConstantKernel(100.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=42, normalize_y=True)

#fir the Gaussian Process Regressor
print('fitting gpr')
gpr.fit(X_train, y_train) 


### GPR with first 15 engines (all rows for respective engines)
# Subset: ALL rows from the first 15 engines (no missing cycles within each engine)
df_first15 = df[df['engine_id'] <= 15]
X_train_2 = df_first15[[f"sensor_{i}" for i in [2,3,4,7,11,12,15,17,20,21]]].values
y_train_2 = df_first15['RUL'].values
print(f"X_train_2 shape: {X_train_2.shape}")   # ~ (2900-3100, 10) depending on those engines' lifespans
# Second GPR (same kernel/settings as your first one, for a fair comparison)
kernel_2 = ConstantKernel(100.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr_2 = GaussianProcessRegressor(kernel=kernel_2, n_restarts_optimizer=5,
                                 random_state=42, normalize_y=True)
gpr_2.fit(X_train_2, y_train_2)


### Random Forest Regressor with all data
#select ALL the data from the df for predicting 
X_train = df[[f"sensor_{i}" for i in [2,3,4,7,11,12,15,17,20,21]]].values  # trending sensors
y_train = df['RUL'].values

#fit the Random Forest Regressor
print('fitting rf')
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)     # all 20k rows, no problem



SENSOR_COLS = [f"sensor_{i}" for i in [2,3,4,7,11,12,15,17,20,21]]

### Load the official test set + ground-truth RUL
file_path = Path(__file__).resolve().parent / "test_FD001.txt"
df_test = pd.read_csv(file_path, sep='\s+', header=None, names=cols)

cols_RUL = ["RUL"]
file_path = Path(__file__).resolve().parent / "RUL_FD001.txt"
df_RUL = pd.read_csv(file_path, sep='\s+', header=None, names=cols_RUL)


### Evaluate gpr, gpr_2 and rf on the official test set
# In test_FD001 each engine is cut off BEFORE failure; RUL_FD001 gives the true RUL
# at each engine's LAST recorded cycle. So we take the last cycle per engine, predict,
# and compare to RUL_FD001 (which is in engine order 1..100).
last_idx = df_test.groupby('engine_id')['cycle'].idxmax()   # row index of max cycle per engine
last_cycles = df_test.loc[last_idx]                          # one row per engine, in engine order
X_test = last_cycles[SENSOR_COLS].values
y_true = df_RUL['RUL'].values                                # true RUL per test engine


### 1D CNN: a sequence model for comparison with the row-wise GPR/RF
# GPR and RF map ONE row (10 sensors) -> one RUL. The CNN instead reads a WINDOW of
# consecutive cycles (WINDOW x 10) and learns the degradation TREND over time.
# This means the data prep is the real work: scale sensors and build sliding windows.
# The RUL label is left uncapped (linear all the way), so engines with longer-than-
# average lifespans keep their full, possibly optimistic, RUL values.
WINDOW = 30          # how many consecutive cycles the CNN sees at once
N_SENSORS = len(SENSOR_COLS)

# Normalize sensors: fit the scaler on TRAIN only, then apply to test (no data leakage)
scaler = MinMaxScaler()
df_cnn = df.copy()
df_cnn[SENSOR_COLS] = scaler.fit_transform(df[SENSOR_COLS])
df_test_cnn = df_test.copy()
df_test_cnn[SENSOR_COLS] = scaler.transform(df_test[SENSOR_COLS])

# Build sliding windows from each TRAIN engine: every block of WINDOW cycles ->
# label = RUL at the last cycle of that window (uncapped, full linear RUL).
def make_train_windows(data, window, sensor_cols):
    X, y = [], []
    for _, eng in data.groupby('engine_id'):
        eng = eng.sort_values('cycle')
        s = eng[sensor_cols].values
        rul = eng['RUL'].values
        for start in range(len(eng) - window + 1):
            X.append(s[start:start + window])
            y.append(rul[start + window - 1])
    return np.array(X), np.array(y)

X_cnn, y_cnn = make_train_windows(df_cnn, WINDOW, SENSOR_COLS)
print(f"CNN training tensor: {X_cnn.shape}  (windows, cycles, sensors)")

# For each TEST engine, take the LAST window (most recent WINDOW cycles). groupby keeps
# engine order 1..100, matching RUL_FD001. Pad engines shorter than WINDOW at the front.
def make_test_windows(data, window, sensor_cols):
    X = []
    for _, eng in data.groupby('engine_id'):
        eng = eng.sort_values('cycle')
        s = eng[sensor_cols].values
        if len(s) >= window:
            X.append(s[-window:])
        else:
            pad = np.repeat(s[:1], window - len(s), axis=0)
            X.append(np.vstack([pad, s]))
    return np.array(X)

X_test_cnn = make_test_windows(df_test_cnn, WINDOW, SENSOR_COLS)

# Define the 1D CNN: stacked Conv1D layers learn local degradation patterns, then
# dense layers regress to a single RUL value.
def build_cnn(window, n_sensors):
    model = keras.Sequential([
        keras.Input(shape=(window, n_sensors)),
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

cnn = build_cnn(WINDOW, N_SENSORS)
cnn.summary()

print('fitting cnn')
early_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
cnn.fit(X_cnn, y_cnn, validation_split=0.1, epochs=50, batch_size=256,
        callbacks=[early_stop], verbose=1)


pred_gpr   = gpr.predict(X_test)
pred_gpr_2 = gpr_2.predict(X_test)
pred_rf    = rf.predict(X_test)
pred_cnn   = cnn.predict(X_test_cnn).ravel()

models = [
    (pred_gpr,   "GPR 1 (3000 random rows)"),
    (pred_gpr_2, "GPR 2 (first 15 engines)"),
    (pred_rf,    "Random Forest (all rows)"),
    (pred_cnn,   "1D CNN (windows)"),
]

# Parity plots: predicted vs true RUL on the test set, one panel per model
fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharex=True, sharey=True)
for ax, (pred, title) in zip(axes, models):
    ax.scatter(y_true, pred, s=20, alpha=0.5)
    lims = [0, max(y_true.max(), pred.max())]
    ax.plot(lims, lims, 'r--', label='perfect (y = x)')
    rmse = np.sqrt(np.mean((pred - y_true) ** 2))
    mae = np.mean(np.abs(pred - y_true))
    ax.set_xlabel("True RUL (RUL_FD001)")
    ax.set_ylabel("Predicted RUL")
    ax.set_title(f"{title}\nRMSE = {rmse:.1f} | MAE = {mae:.1f} cycles")
    ax.legend(); ax.grid(alpha=0.3)
fig.suptitle("Test-set RUL prediction vs. RUL_FD001 (last cycle per test engine)", fontweight="bold")
fig.tight_layout()
save_figure("fd001_parity.png")
maybe_show()


# Numeric comparison: collect every model's metrics into one tidy summary table,
# sorted best-first by RMSE. A single comparable table is far more readable (and
# screenshot-friendly for a write-up) than scattered prints.
# PBF (Predicted Before Failure): % of test engines where predicted RUL is LOWER than
# the true RUL — i.e. the model expects failure sooner than reality, so maintenance would
# be scheduled before the real failure. Higher = safer/more conservative.
results = pd.DataFrame(
    [{'Model': title,
      'RMSE': np.sqrt(np.mean((pred - y_true) ** 2)),
      'MAE': np.mean(np.abs(pred - y_true)),
      'PBF (%)': np.mean(pred < y_true) * 100}
     for pred, title in models]
).sort_values('RMSE').reset_index(drop=True)

print("\nTest-set performance (vs RUL_FD001):")
print(results.round(2).to_string(index=False))


### Relative-error view: per-engine error normalised by that engine's TOTAL lifespan
# RMSE/MAE are absolute cycles, which is hard to judge without knowing how long an
# engine lives. A 20-cycle miss is small for a 300-cycle engine but large for a
# 100-cycle one. So we express each engine's error as a fraction of its full life:
#   total_life = last recorded cycle + true RUL (RUL_FD001)   # the engine's whole run
#   rel_error  = |pred - true_RUL| / total_life
# Then we sort engines from smallest to largest ratio to see the spread of relative error.
total_life = last_cycles['cycle'].values + y_true          # full lifespan per test engine
print(f"\nTest-engine lifespan: min={total_life.min():.0f}, "
      f"median={np.median(total_life):.0f}, max={total_life.max():.0f} cycles")

# ECDF: x = relative error (% of total lifespan), y = cumulative % of engines.
# Read it as "Y% of engines are within X% error" — e.g. where each curve crosses the
# dashed 90% line tells you the error threshold that covers 90% of engines.
colors = {'GPR 1 (3000 random rows)': 'tab:blue',
          'GPR 2 (first 15 engines)': 'tab:green',
          'Random Forest (all rows)': 'tab:orange',
          '1D CNN (windows)': 'tab:red'}
n = len(y_true)
ecdf_y = np.arange(1, n + 1) / n * 100        # cumulative percentage of engines

plt.figure(figsize=(11, 6))
for pred, title in models:
    rel_error = np.abs(pred - y_true) / total_life * 100   # percent of total lifespan
    rel_sorted = np.sort(rel_error)
    median_rel = np.median(rel_error)
    p90 = np.percentile(rel_error, 90)
    plt.step(rel_sorted, ecdf_y, where='post', color=colors[title], lw=2,
             label=f"{title}  (median {median_rel:.1f}% | 90% within {p90:.1f}%)")
plt.axhline(90, color='grey', ls='--', lw=1, alpha=0.7)
plt.text(plt.xlim()[1], 90, ' 90% of engines', va='bottom', ha='right',
         color='grey', fontsize=9)
plt.xlabel("Relative RUL error: |pred − true| / total lifespan (%)")
plt.ylabel("Cumulative % of engines within that error")
plt.title("How tightly each model predicts RUL (ECDF of relative error)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
save_figure("fd001_ecdf_relative_error.png")
maybe_show()


# For the trajectory plots the CNN needs ONE prediction per cycle. We build a window
# ending at each cycle (the most recent WINDOW cycles), padding the front for early
# cycles, exactly like make_test_windows but evaluated at every cycle of the engine.
def make_per_cycle_windows(s, window):
    X = []
    for i in range(len(s)):
        w = s[max(0, i - window + 1):i + 1]
        if len(w) < window:
            pad = np.repeat(w[:1], window - len(w), axis=0)
            w = np.vstack([pad, w])
        X.append(w)
    return np.array(X)


### Trajectory plot: true RUL (linear) vs all-model predictions for one test engine
example_engine = 1   # first test engine (held out for all models)
eng = df_test[df_test['engine_id'] == example_engine].sort_values('cycle')
X_eng = eng[SENSOR_COLS].values
cycles = eng['cycle'].values
# test engines are cut off before failure: RUL_FD001 is the true RUL at the LAST cycle.
# Reconstruct the full linear RUL line: at earlier cycles RUL is higher by (last_cycle - cycle).
final_rul = df_RUL['RUL'].iloc[example_engine - 1]
true_rul = final_rul + (cycles.max() - cycles)

p1, s1 = gpr.predict(X_eng, return_std=True)
p2, s2 = gpr_2.predict(X_eng, return_std=True)
pred_rf_eng = rf.predict(X_eng)
# CNN uses the SCALED sensors and per-cycle windows (same scaler fit on train)
s_eng = df_test_cnn[df_test_cnn['engine_id'] == example_engine].sort_values('cycle')[SENSOR_COLS].values
pred_cnn_eng = cnn.predict(make_per_cycle_windows(s_eng, WINDOW)).ravel()

plt.figure(figsize=(10, 6))
plt.plot(cycles, true_rul, 'k-', lw=2, label='True RUL (linear)')
plt.plot(cycles, p1, 'b-', label='GPR 1 (3000 random rows)')
plt.fill_between(cycles, p1 - 1.96*s1, p1 + 1.96*s1, color='blue', alpha=0.15)
plt.plot(cycles, p2, 'g-', label='GPR 2 (first 15 engines)')
plt.fill_between(cycles, p2 - 1.96*s2, p2 + 1.96*s2, color='green', alpha=0.15)
plt.plot(cycles, pred_rf_eng, color='tab:orange', label='Random Forest (all rows)')
plt.plot(cycles, pred_cnn_eng, color='tab:red', label='1D CNN (windows)')
plt.xlabel("Cycle"); plt.ylabel("RUL (cycles remaining)")
plt.title(f"RUL prediction vs. true — test engine {example_engine}")
plt.legend(); plt.grid(alpha=0.3)
save_figure("fd001_trajectory_test.png")
maybe_show()


### Trajectory plot on a TRAIN engine: full run-to-failure (RUL reaches 0)
# Unlike the test set, train_FD001 contains every cycle up to failure, so the true
# RUL is a clean line that decreases all the way down to 0. We pick one engine at
# random and overlay the GPR predictions + 95% bands, just like the test-engine plot.
rng = np.random.default_rng(42)
train_engine = int(rng.choice(df['engine_id'].unique()))
eng_tr = df[df['engine_id'] == train_engine].sort_values('cycle')
X_eng_tr = eng_tr[SENSOR_COLS].values
cycles_tr = eng_tr['cycle'].values
true_rul_tr = eng_tr['RUL'].values   # already linear down to 0 at the final (failure) cycle

p1_tr, s1_tr = gpr.predict(X_eng_tr, return_std=True)
p2_tr, s2_tr = gpr_2.predict(X_eng_tr, return_std=True)
pred_rf_tr = rf.predict(X_eng_tr)
# CNN uses the SCALED sensors and per-cycle windows
s_eng_tr = df_cnn[df_cnn['engine_id'] == train_engine].sort_values('cycle')[SENSOR_COLS].values
pred_cnn_tr = cnn.predict(make_per_cycle_windows(s_eng_tr, WINDOW)).ravel()

plt.figure(figsize=(10, 6))
plt.plot(cycles_tr, true_rul_tr, 'k-', lw=2, label='True RUL (run-to-failure)')
plt.plot(cycles_tr, p1_tr, 'b-', label='GPR 1 (3000 random rows)')
plt.fill_between(cycles_tr, p1_tr - 1.96*s1_tr, p1_tr + 1.96*s1_tr, color='blue', alpha=0.15)
plt.plot(cycles_tr, p2_tr, 'g-', label='GPR 2 (first 15 engines)')
plt.fill_between(cycles_tr, p2_tr - 1.96*s2_tr, p2_tr + 1.96*s2_tr, color='green', alpha=0.15)
plt.plot(cycles_tr, pred_rf_tr, color='tab:orange', label='Random Forest (all rows)')
plt.plot(cycles_tr, pred_cnn_tr, color='tab:red', label='1D CNN (windows)')
plt.xlabel("Cycle"); plt.ylabel("RUL (cycles remaining)")
plt.title(f"RUL prediction vs. true — train engine {train_engine} (full run-to-failure)")
plt.legend(); plt.grid(alpha=0.3)
save_figure("fd001_trajectory_train.png")
maybe_show()
