#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Height–Diameter (H–D) modelling for Quercus robur

Two-step strategy for each model:
  (1) Fit per plot (parcella) to assess variability + get robust initial values
  (2) Global fit on full dataset using robust p0 from step (1)

Then: PER-PLOT CALIBRATION of the GLOBAL model
  For each plot, estimate a single scale factor a_plot:
     h_cal(d) = h0 + a_plot * (h_global(d) - h0)
  a_plot is estimated by least squares within each plot.

Models (M1–M5):
M1  Chapman-Richards:  h = 1.3 + β0 * (1 - e^{-β1 d})^{β2}
M2  Logistic:          h = 1.3 + β0 / (1 + β1 * e^{-β2 d})
M3  Naslund:           h = 1.3 + d^2 / (β0 + β1 d)^2
M4  Weibull:           h = 1.3 + β0 * (1 - e^{-β1 d^{β2}})
M5  Wykoff:            h = 1.3 + exp(β0 + β1 / (d + 1))

Outputs:
STEP (1) per-plot:
- CSV: hd_params_by_plot_M1_to_M5.csv
- CSV: hd_robust_p0_from_plots_M1_to_M5.csv

STEP (2) global:
- Plot: hd_global_M1_to_M5.png
- CSV:  hd_params_M1_to_M5.csv
- Diagnostics:
    * Q–Q residuals (overlay):                hd_diagnostics/qq_residuals_M1_to_M5.png
    * Residuals vs Fitted (ONE figure):       hd_diagnostics/resid_vs_fitted_M1_to_M5.png

CALIBRATION (post global):
- CSV:  hd_calibration_scale_by_plot_M1_to_M5.csv
- CSV:  hd_metrics_global_calibrated_M1_to_M5.csv
- CSV:  hd_metrics_by_plot_calibrated_M1_to_M5.csv
- Plots: hd_calibration/obs_vs_pred_<MODELKEY>.png

FIGURE SOLO M2:
- Plot: hd_calibration/hd_M2_global_vs_calibrated.png
  (stesso colore; linea continua=global, tratteggiata=calibrata)

SPARSE CALIBRATION EVALUATION (operational):
- CSV:  hd_sparse_calibration_eval/hd_sparsecal_long_M1_to_M5.csv
- CSV:  hd_sparse_calibration_eval/hd_sparsecal_summary_M1_to_M5.csv
- Plot: hd_sparse_calibration_eval/hd_sparsecal_rmse_vs_ncal.png

LOPO (Leave-One-Plot-Out) generalisation:
- CSV:  hd_lopo_eval/hd_lopo_long_M1_to_M5.csv
- CSV:  hd_lopo_eval/hd_lopo_summary_M1_to_M5.csv
- Plot: hd_lopo_eval/hd_lopo_rmse_vs_ncal.png

BOOTSTRAP (stability / robustness):
- CSV (long):    hd_bootstrap_long_M1_to_M5_<MODE>_B<B>.csv
- CSV (summary): hd_bootstrap_summary_M1_to_M5_<MODE>_B<B>.csv
  where MODE in {"within_plot", "flat", "cluster_plot"}.
  Recommended: MODE="within_plot" (stratified within plots).

PRINTS (solo):
- Global metrics without calibration
- Global metrics with calibration
- Sparse calibration summary (if enabled)
- LOPO summary (if enabled)
- Bootstrap summary (if enabled)
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats
from matplotlib.ticker import MultipleLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:
    lowess = None


# =========================================================
# PRINT CONTROL (keep prints exactly as requested in docstring)
# =========================================================
PRINT_EXTRA_STATUS = False   # keep False to avoid "Saved ..." messages etc.


# =========================================================
# BOOTSTRAP CONFIG
# =========================================================
DO_BOOTSTRAP = True
BOOT_B = 500                 # e.g., 200–500 quick; 1000 for more stable CIs
BOOT_MODE = "within_plot"    # "within_plot" (recommended), "flat", "cluster_plot"
BOOT_SEED = 123
BOOT_DO_CAL = True           # include calibrated metrics in bootstrap


# =========================================================
# OPERATIONAL / SPARSE CALIBRATION EVALUATION
#   Calibra a_plot usando SOLO n_cal alberi per parcella,
#   valuta su alberi rimanenti (hold-out within plot).
# =========================================================
DO_SPARSE_CAL_EVAL = True
SPARSE_CAL_N_LIST = [1, 3, 5]     # alberi per parcella usati per stimare a_plot
SPARSE_CAL_REPS = 300             # ripetizioni random (200–500 è ok)
SPARSE_CAL_SEED = 2025
SPARSE_A_MIN, SPARSE_A_MAX = 0.1, 5.0
SPARSE_DIR = "hd_sparse_calibration_eval"
os.makedirs(SPARSE_DIR, exist_ok=True)


# =========================================================
# LOPO (Leave-One-Plot-Out) EVALUATION (generalisation to new plots)
#   Per ogni plot p:
#     - fitta il modello globale sui plot != p
#     - valuta su p
#     - opzionale: sparse calibration su p usando solo n_cal alberi (operational)
# =========================================================
DO_LOPO_EVAL = True
LOPO_N_LIST = [1, 3, 5]          # alberi del plot held-out usati per stimare a_plot
LOPO_REPS = 200                  # ripetizioni random per la scelta dei n_cal
LOPO_SEED = 4242
LOPO_A_MIN, LOPO_A_MAX = 0.1, 5.0
LOPO_DIR = "hd_lopo_eval"
os.makedirs(LOPO_DIR, exist_ok=True)

# se True: p0 per ciascun fold è calcolato ESCLUDENDO il plot hold-out (no leakage)
LOPO_STRICT_P0 = True


def _status(msg: str) -> None:
    if PRINT_EXTRA_STATUS:
        print(msg)


# -------------------------
# Plot styling helpers
# -------------------------
def stylize_axes(ax, xlabel=None, ylabel=None, label_size=16, tick_size=13, spine_lw=1.6):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(spine_lw)
    ax.spines["left"].set_linewidth(spine_lw)
    ax.tick_params(axis="both", which="both", labelsize=tick_size, width=spine_lw)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size, fontweight="bold")


# -------------------------
# Fixed colors (consistent across ALL plots)
# -------------------------
_tab10 = plt.get_cmap("tab10").colors
COLORS = {
    "M1_ChapmanRichards": _tab10[0],
    "M2_Logistic":        _tab10[1],
    "M3_Naslund":         _tab10[2],
    "M4_Weibull":         _tab10[3],
    "M5_Wykoff":          _tab10[4],
}
pretty = {
    "M1_ChapmanRichards": "M1 Chapman-Richards",
    "M2_Logistic":        "M2 Logistic",
    "M3_Naslund":         "M3 Naslund",
    "M4_Weibull":         "M4 Weibull",
    "M5_Wykoff":          "M5 Wykoff",
}


# -------------------------
# 0. I/O and data loading
# -------------------------
CANDIDATE_FILES = ["Date H-D_new.xlsx", "Date H-D.xlsx", "Date H-D.xls", "Date H-D .xlsx"]


def smart_read():
    for fn in CANDIDATE_FILES:
        if os.path.exists(fn):
            try:
                df_ = pd.read_excel(fn, header=1)
            except Exception:
                df_ = pd.read_excel(fn)
            return df_, fn
    raise FileNotFoundError(f"None of the expected files found: {CANDIDATE_FILES}")


raw_df, used_file = smart_read()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "parcela": "Plot",
        "arbore_id": "Tree_ID",
        "arbore id": "Tree_ID",
        "diametru,cm": "Diameter_cm",
        "diametru, cm": "Diameter_cm",
        "diameter_cm": "Diameter_cm",
        "înălţime, m": "Height_m",
        "înălțime, m": "Height_m",
        "inaltime, m": "Height_m",
        "height_m": "Height_m",
        "g_i": "G_i",
        "dq_i": "Dq_i",
        "hdom": "Hdom",
        "pcor": "Pcor",
        "dcor": "Dcor",
        "h/d": "H_D",
        "h_d": "H_D",
        "helag": "Helag",
    }

    # some exports include leading unnamed columns
    if df.columns[:2].tolist() == ["Unnamed: 0", "Unnamed: 1"]:
        df = df.iloc[:, 2:].copy()

    # some exports have a header row duplicated as first row
    if df.shape[0] > 1 and (df.iloc[0].isna().sum() <= 3) and str(df.columns[0]).startswith("Unnamed"):
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

    new_cols = []
    for c in df.columns:
        key = str(c).strip().lower()
        new_cols.append(mapping.get(key, c))
    df.columns = new_cols

    # fallback if columns not matched
    if not set(["Plot", "Tree_ID", "Diameter_cm", "Height_m"]).issubset(set(df.columns)):
        if df.shape[1] >= 4:
            df = df.rename(columns={
                df.columns[0]: "Plot",
                df.columns[1]: "Tree_ID",
                df.columns[2]: "Diameter_cm",
                df.columns[3]: "Height_m",
            })
    return df


df = normalize_columns(raw_df)

known_cols = ["Plot", "Tree_ID", "Diameter_cm", "Height_m", "G_i", "Dq_i", "Hdom", "Pcor", "Dcor", "H_D", "Helag"]
keep = [c for c in known_cols if c in df.columns]
df = df[keep].copy()

for c in ["Diameter_cm", "Height_m", "G_i", "Dq_i", "Hdom", "Pcor", "Dcor", "H_D", "Helag"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Plot", "Diameter_cm", "Height_m"]).copy()
df["Plot"] = df["Plot"].astype(str)
df = df[(df["Diameter_cm"] > 0) & (df["Height_m"] > 0)].reset_index(drop=True)


# -------------------------
# 1. Data arrays & helpers
# -------------------------
h0 = 1.3
D_all = df["Diameter_cm"].to_numpy(dtype=float)
H_all = df["Height_m"].to_numpy(dtype=float)
G_all = df["Plot"].to_numpy(dtype=str)
plots = sorted(df["Plot"].unique().tolist())
n_plots = len(plots)

H_top = np.nanpercentile(H_all, 95.0)
amp_guess = max(H_top - h0, 5.0)
rate_guess = 0.05


def metrics(y_obs, y_pred, k):
    """
    RMSE (df-adjusted) + R2_adj + Bias + RSS + AIC.
    AIC uses: n*log(RSS/n) + 2*(k+1)  (Gaussian, constant omitted).
    """
    y_obs = np.asarray(y_obs, float)
    y_pred = np.asarray(y_pred, float)
    n = int(y_obs.size)

    rss = float(np.nansum((y_obs - y_pred) ** 2))
    denom = max(n - k - 1, 1)
    rmse = float(np.sqrt(rss / denom))

    tss = float(np.nansum((y_obs - np.nanmean(y_obs)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    r2adj = 1.0 - (1.0 - r2) * (n - 1) / denom if np.isfinite(r2) else np.nan

    bias = float(np.nanmean(y_pred - y_obs))

    aic = float(n * np.log(max(rss / max(n, 1), 1e-12)) + 2.0 * (k + 1))

    return dict(RMSE=rmse, R2_adj=r2adj, Bias=bias, RSS=rss, AIC=aic)


def metrics_plain(y_obs, y_pred):
    """Plain predictive metrics (no df-adjustment)."""
    y_obs = np.asarray(y_obs, float)
    y_pred = np.asarray(y_pred, float)
    n = int(y_obs.size)

    rss = float(np.nansum((y_obs - y_pred) ** 2))
    rmse = float(np.sqrt(rss / max(n, 1)))
    tss = float(np.nansum((y_obs - np.nanmean(y_obs)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    bias = float(np.nanmean(y_pred - y_obs))

    return dict(RMSE=rmse, R2=r2, Bias=bias, RSS=rss)


def _safe_curve_fit(func, D, H, p0, bounds, maxfev=60000):
    """Return (popt, success, err_msg)."""
    try:
        popt, _ = curve_fit(func, D, H, p0=p0, bounds=bounds, maxfev=maxfev)
        if not np.all(np.isfinite(popt)):
            return np.full(len(p0), np.nan, float), False, "non-finite popt"
        return popt, True, ""
    except Exception as e:
        return np.full(len(p0), np.nan, float), False, str(e)


def _near_bounds(p, lo, hi, rel=0.01, abs_tol=1e-8):
    p = np.asarray(p, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    rng = np.maximum(hi - lo, abs_tol)
    return np.any((p - lo) <= rel * rng) or np.any((hi - p) <= rel * rng)


# -------------------------
# 2. Model kernels (M1–M5)
# -------------------------
def m1_chapman_richards(D, beta0, beta1, beta2):
    D = np.asarray(D, float)
    base = np.clip(1.0 - np.exp(-np.clip(beta1, 1e-9, None) * D), 1e-12, 1.0)
    return h0 + np.clip(beta0, 1e-9, None) * (base ** np.clip(beta2, 1e-9, None))


def m2_logistic(D, beta0, beta1, beta2):
    D = np.asarray(D, float)
    return h0 + np.clip(beta0, 1e-9, None) / (
        1.0 + np.clip(beta1, 1e-9, None) * np.exp(-np.clip(beta2, 1e-9, None) * D)
    )


def m3_naslund(D, beta0, beta1):
    D = np.asarray(D, float)
    denom = np.clip(beta0 + beta1 * D, 1e-9, None)
    return h0 + (D ** 2) / (denom ** 2)


def m4_weibull(D, beta0, beta1, beta2):
    D = np.asarray(D, float)
    return h0 + np.clip(beta0, 1e-9, None) * (
        1.0 - np.exp(-np.clip(beta1, 1e-9, None) * np.power(D, np.clip(beta2, 1e-9, None)))
    )


def m5_wykoff(D, beta0, beta1):
    D = np.asarray(D, float)
    inside = beta0 + beta1 / (1.0 + D)
    inside = np.clip(inside, -10.0, 5.0)
    return h0 + np.exp(inside)


# -------------------------
# 3. Model specs (bounds, p0 base, k params)
# -------------------------
A_LO, A_HI = 3.0, 60.0
R_LO, R_HI = 1e-4, 1.0
S_LO, S_HI = 0.3, 6.0

MODEL_SPECS = {
    "M1_ChapmanRichards": dict(
        func=m1_chapman_richards,
        k=3,
        bounds=([A_LO, R_LO, S_LO], [A_HI, R_HI, S_HI]),
        p0_fallback=[amp_guess, rate_guess, 1.5],
    ),
    "M2_Logistic": dict(
        func=m2_logistic,
        k=3,
        bounds=([A_LO, 0.1, R_LO], [A_HI, 100.0, R_HI]),
        p0_fallback=[amp_guess, 1.0, rate_guess],
    ),
    "M3_Naslund": dict(
        func=m3_naslund,
        k=2,
        bounds=([1.0, R_LO], [100.0, R_HI]),
        p0_fallback=[10.0, 0.05],
    ),
    "M4_Weibull": dict(
        func=m4_weibull,
        k=3,
        bounds=([A_LO, R_LO, 0.5], [A_HI, R_HI, 5.0]),
        p0_fallback=[amp_guess, rate_guess, 1.5],
    ),
    "M5_Wykoff": dict(
        func=m5_wykoff,
        k=2,
        bounds=([-2.0, -50.0], [5.0, 50.0]),
        p0_fallback=[3.0, 5.0],
    ),
}
model_order = list(MODEL_SPECS.keys())


# =========================================================
# STEP (1) — Fit each model per plot
# =========================================================
by_plot_rows = []

for plot_id in plots:
    dfi = df[df["Plot"] == plot_id]
    Di = dfi["Diameter_cm"].to_numpy(float)
    Hi = dfi["Height_m"].to_numpy(float)
    n_i = int(Hi.size)

    for mname in model_order:
        spec = MODEL_SPECS[mname]
        func = spec["func"]
        k = spec["k"]
        lo, hi = spec["bounds"]

        if n_i < (k + 2):
            by_plot_rows.append({
                "Plot": plot_id, "Model": mname, "n": n_i,
                "success": False, "err": "too few points",
                **{f"p{i+1}": np.nan for i in range(k)},
                **{kk: np.nan for kk in ["RMSE", "R2_adj", "Bias", "RSS", "AIC"]},
            })
            continue

        p0 = spec["p0_fallback"]
        popt, ok, err = _safe_curve_fit(func, Di, Hi, p0=p0, bounds=(lo, hi), maxfev=60000)

        if ok:
            yhat = func(Di, *popt)
            met = metrics(Hi, yhat, k=k)
        else:
            met = {kk: np.nan for kk in ["RMSE", "R2_adj", "Bias", "RSS", "AIC"]}

        row = {"Plot": plot_id, "Model": mname, "n": n_i, "success": bool(ok), "err": err if not ok else ""}
        for i in range(k):
            row[f"p{i+1}"] = float(popt[i]) if ok else np.nan
        row.update({kk: float(met[kk]) if np.isfinite(met[kk]) else np.nan for kk in met})
        by_plot_rows.append(row)

by_plot_df = pd.DataFrame(by_plot_rows)
BY_PLOT_CSV = "hd_params_by_plot_M1_to_M5.csv"
by_plot_df.to_csv(BY_PLOT_CSV, index=False)

# ---- Robust p0 from per-plot fits (median, excluding near bounds)
robust_p0_rows = []
robust_p0_map = {}

for mname in model_order:
    spec = MODEL_SPECS[mname]
    k = spec["k"]
    lo, hi = map(np.asarray, spec["bounds"])

    sub = by_plot_df[(by_plot_df["Model"] == mname) & (by_plot_df["success"] == True)].copy()
    robust_p0 = None

    if not sub.empty:
        P = sub[[f"p{i+1}" for i in range(k)]].to_numpy(float)
        P = P[np.isfinite(P).all(axis=1)]
        P2 = [p for p in P if not _near_bounds(p, lo, hi, rel=0.01)]
        P2 = np.asarray(P2, float) if len(P2) else np.empty((0, k))

        if P2.shape[0] >= 2:
            robust_p0 = np.nanmedian(P2, axis=0)
            used = "median_plots_excl_bounds"
        elif P.shape[0] >= 2:
            robust_p0 = np.nanmedian(P, axis=0)
            used = "median_plots_all"
        else:
            used = "fallback"
    else:
        used = "fallback"

    if robust_p0 is None or (not np.all(np.isfinite(robust_p0))):
        robust_p0 = np.asarray(spec["p0_fallback"], float)

    eps = 1e-6
    robust_p0 = np.minimum(np.maximum(robust_p0, lo + eps), hi - eps)

    robust_p0_map[mname] = robust_p0.tolist()
    row = {"Model": mname, "p0_source": used}
    for i in range(k):
        row[f"p0_{i+1}"] = float(robust_p0[i])
    robust_p0_rows.append(row)

robust_p0_df = pd.DataFrame(robust_p0_rows)
ROBUST_P0_CSV = "hd_robust_p0_from_plots_M1_to_M5.csv"
robust_p0_df.to_csv(ROBUST_P0_CSV, index=False)


# =========================================================
# STEP (2) — Global fits using robust p0
# =========================================================
results = {}


def fit_model_global(name, func, p0, bounds, k_params):
    lo, hi = bounds
    popt, ok, err = _safe_curve_fit(func, D_all, H_all, p0=p0, bounds=(lo, hi), maxfev=60000)

    if ok:
        yhat = func(D_all, *popt)
        m = metrics(H_all, yhat, k=k_params)
    else:
        yhat = np.full_like(H_all, np.nan)
        m = {kk: np.nan for kk in ["RMSE", "R2_adj", "Bias", "RSS", "AIC"]}

    results[name] = dict(
        p=popt,
        metrics=m,
        ok=ok,
        err=err,
        yhat=yhat,
        predict=lambda D, _p=popt, _f=func: _f(D, *_p) if np.all(np.isfinite(_p)) else np.full_like(np.asarray(D, float), np.nan),
    )
    return popt, ok


for mname in model_order:
    spec = MODEL_SPECS[mname]
    p0 = robust_p0_map.get(mname, spec["p0_fallback"])
    fit_model_global(mname, spec["func"], p0=p0, bounds=spec["bounds"], k_params=spec["k"])


# -------------------------
# 4. Plot: global curves (M1–M5) — ONE axes
# -------------------------
diam_grid = np.linspace(D_all.min(), D_all.max(), 400)
fig, ax = plt.subplots(figsize=(10.8, 6.6))
ax.scatter(D_all, H_all, alpha=0.22, s=12, label="Observed", color="0.5")

for name in model_order:
    if not results[name]["ok"]:
        continue
    y = results[name]["predict"](diam_grid)
    ax.plot(diam_grid, y, label=pretty[name], linewidth=3.2, color=COLORS[name])

stylize_axes(ax, xlabel="Diameter (cm)", ylabel="Height (m)", label_size=18, tick_size=14, spine_lw=1.7)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.legend(ncol=2, fontsize=10, frameon=False)
fig.tight_layout()
PNG_OUT = "hd_global_M1_to_M5.png"
fig.savefig(PNG_OUT, dpi=160)
plt.close(fig)


# -------------------------
# 5. Save global params + metrics (and PRINT ONLY METRICS)
# -------------------------
rows = []
for name in model_order:
    res = results[name]
    spec = MODEL_SPECS[name]
    k = spec["k"]
    row = {"Model": name, "success": bool(res.get("ok", False)), "err": res.get("err", "")}
    p = res.get("p", None)
    for i in range(k):
        row[f"p{i+1}"] = float(p[i]) if (p is not None and np.isfinite(p[i])) else np.nan
    for kk, vv in res.get("metrics", {}).items():
        row[kk] = float(vv) if np.isfinite(vv) else np.nan
    rows.append(row)

out_df = pd.DataFrame(rows)
CSV_OUT = "hd_params_M1_to_M5.csv"
out_df.to_csv(CSV_OUT, index=False)

metrics_uncal = out_df[["Model", "RMSE", "R2_adj", "Bias", "RSS", "AIC"]].copy().sort_values("Model")

print("\nGLOBAL METRICS (NO calibration):")
with pd.option_context("display.max_columns", None, "display.width", 160):
    print(metrics_uncal.to_string(index=False))


# =========================================================
# CALIBRATION — compute a_plot for each model and plot
# =========================================================
CAL_DIR = "hd_calibration"
os.makedirs(CAL_DIR, exist_ok=True)


def calibrate_scales_by_plot(yhat_global, H, G, h0=1.3, a_min=0.1, a_max=5.0):
    """
    Fit a_plot for each plot:
        H - h0 ≈ a_plot * (yhat_global - h0)
    a_plot = sum(g*t) / sum(g^2), where g=(yhat_global-h0), t=(H-h0).
    """
    g_all = np.asarray(yhat_global, float) - h0
    t_all = np.asarray(H, float) - h0

    uniq = np.unique(G)
    out = {}
    diag = {}

    for pl in uniq:
        mask = (G == pl)
        g = g_all[mask]
        t = t_all[mask]

        denom = float(np.sum(g * g))
        num = float(np.sum(g * t))

        if not np.isfinite(denom) or denom <= 1e-12 or not np.isfinite(num):
            a = np.nan
        else:
            a = num / denom

        if np.isfinite(a):
            a = float(np.clip(a, a_min, a_max))

        out[pl] = a
        diag[pl] = dict(num=num, denom=denom, n=int(np.sum(mask)))

    return out, diag


# =========================================================
# SPARSE CALIBRATION HELPERS / EVAL
# =========================================================
def _fit_a_from_subset(yhat_subset: np.ndarray, h_subset: np.ndarray,
                       h0: float = 1.3, a_min: float = 0.1, a_max: float = 5.0) -> float:
    g = np.asarray(yhat_subset, float) - h0
    t = np.asarray(h_subset, float) - h0
    denom = float(np.sum(g * g))
    num = float(np.sum(g * t))
    if (not np.isfinite(num)) or (not np.isfinite(denom)) or denom <= 1e-12:
        return 1.0  # neutral fallback
    a = num / denom
    if np.isfinite(a):
        a = float(np.clip(a, a_min, a_max))
    else:
        a = 1.0
    return a


def _predict_calibrated_with_a(yhat: np.ndarray, a: float, h0: float = 1.3) -> np.ndarray:
    yhat = np.asarray(yhat, float)
    return h0 + float(a) * (yhat - h0)


def sparse_calibration_eval_for_model(
    mname: str,
    n_list: list[int],
    reps: int,
    seed: int,
    h0: float = 1.3,
    a_min: float = 0.1,
    a_max: float = 5.0,
) -> pd.DataFrame:
    if (mname not in results) or (not results[mname]["ok"]):
        return pd.DataFrame()

    yhat_all = results[mname]["yhat"]
    if not np.all(np.isfinite(yhat_all)):
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    rows = []
    idx_by_plot = {pl: np.where(G_all == pl)[0] for pl in plots}

    for rep in range(reps):
        for n_cal in n_list:
            y_obs_test_all = []
            y_pred_test_all = []
            y_pred_test_uncal_all = []
            plots_used = 0
            n_test_total = 0

            for pl in plots:
                idx_pl = idx_by_plot[pl]
                n_pl = idx_pl.size
                if n_pl <= n_cal:
                    continue

                cal_idx = rng.choice(idx_pl, size=n_cal, replace=False)
                test_idx = np.setdiff1d(idx_pl, cal_idx, assume_unique=False)

                a = _fit_a_from_subset(
                    yhat_subset=yhat_all[cal_idx],
                    h_subset=H_all[cal_idx],
                    h0=h0,
                    a_min=a_min,
                    a_max=a_max
                )

                y_test_cal = _predict_calibrated_with_a(yhat_all[test_idx], a=a, h0=h0)
                y_test_uncal = yhat_all[test_idx]

                y_obs_test_all.append(H_all[test_idx])
                y_pred_test_all.append(y_test_cal)
                y_pred_test_uncal_all.append(y_test_uncal)

                plots_used += 1
                n_test_total += test_idx.size

            if plots_used == 0:
                continue

            y_obs_test_all = np.concatenate(y_obs_test_all)
            y_pred_test_all = np.concatenate(y_pred_test_all)
            y_pred_test_uncal_all = np.concatenate(y_pred_test_uncal_all)

            met_cal = metrics_plain(y_obs_test_all, y_pred_test_all)
            met_unc = metrics_plain(y_obs_test_all, y_pred_test_uncal_all)

            # ---- include Bias + RSS in long rows
            rows.append({
                "Model": mname,
                "rep": int(rep),
                "n_cal": int(n_cal),
                "plots_used": int(plots_used),
                "n_test_total": int(n_test_total),

                "RMSE_test_cal": float(met_cal["RMSE"]),
                "R2_test_cal": float(met_cal["R2"]),
                "Bias_test_cal": float(met_cal["Bias"]),
                "RSS_test_cal": float(met_cal["RSS"]),

                "RMSE_test_uncal": float(met_unc["RMSE"]),
                "R2_test_uncal": float(met_unc["R2"]),
                "Bias_test_uncal": float(met_unc["Bias"]),
                "RSS_test_uncal": float(met_unc["RSS"]),
            })

    return pd.DataFrame(rows)


def summarize_sparse_eval(df_long: pd.DataFrame) -> pd.DataFrame:
    # ---- summary now includes RMSE + Bias + RSS (and deltas)
    if df_long.empty:
        return pd.DataFrame()

    def q(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.quantile(v, 0.025)),
                float(np.quantile(v, 0.50)),
                float(np.quantile(v, 0.975)))

    out = []
    for (m, n_cal), sub in df_long.groupby(["Model", "n_cal"]):
        # RMSE
        rmse_cal = sub["RMSE_test_cal"].to_numpy(float)
        rmse_unc = sub["RMSE_test_uncal"].to_numpy(float)
        rmse_del = rmse_cal - rmse_unc

        # Bias
        bias_cal = sub["Bias_test_cal"].to_numpy(float)
        bias_unc = sub["Bias_test_uncal"].to_numpy(float)
        bias_del = bias_cal - bias_unc

        # RSS
        rss_cal = sub["RSS_test_cal"].to_numpy(float)
        rss_unc = sub["RSS_test_uncal"].to_numpy(float)
        rss_del = rss_cal - rss_unc

        rmse_cal_ci = q(rmse_cal); rmse_unc_ci = q(rmse_unc); rmse_del_ci = q(rmse_del)
        bias_cal_ci = q(bias_cal); bias_unc_ci = q(bias_unc); bias_del_ci = q(bias_del)
        rss_cal_ci  = q(rss_cal);  rss_unc_ci  = q(rss_unc);  rss_del_ci  = q(rss_del)

        out.append({
            "Model": m,
            "n_cal": int(n_cal),

            "RMSE_test_cal_ci2p5": rmse_cal_ci[0],
            "RMSE_test_cal_median": rmse_cal_ci[1],
            "RMSE_test_cal_ci97p5": rmse_cal_ci[2],
            "RMSE_test_uncal_ci2p5": rmse_unc_ci[0],
            "RMSE_test_uncal_median": rmse_unc_ci[1],
            "RMSE_test_uncal_ci97p5": rmse_unc_ci[2],
            "Delta_RMSE_(cal-uncal)_ci2p5": rmse_del_ci[0],
            "Delta_RMSE_(cal-uncal)_median": rmse_del_ci[1],
            "Delta_RMSE_(cal-uncal)_ci97p5": rmse_del_ci[2],

            "Bias_test_cal_ci2p5": bias_cal_ci[0],
            "Bias_test_cal_median": bias_cal_ci[1],
            "Bias_test_cal_ci97p5": bias_cal_ci[2],
            "Bias_test_uncal_ci2p5": bias_unc_ci[0],
            "Bias_test_uncal_median": bias_unc_ci[1],
            "Bias_test_uncal_ci97p5": bias_unc_ci[2],
            "Delta_Bias_(cal-uncal)_ci2p5": bias_del_ci[0],
            "Delta_Bias_(cal-uncal)_median": bias_del_ci[1],
            "Delta_Bias_(cal-uncal)_ci97p5": bias_del_ci[2],

            "RSS_test_cal_ci2p5": rss_cal_ci[0],
            "RSS_test_cal_median": rss_cal_ci[1],
            "RSS_test_cal_ci97p5": rss_cal_ci[2],
            "RSS_test_uncal_ci2p5": rss_unc_ci[0],
            "RSS_test_uncal_median": rss_unc_ci[1],
            "RSS_test_uncal_ci97p5": rss_unc_ci[2],
            "Delta_RSS_(cal-uncal)_ci2p5": rss_del_ci[0],
            "Delta_RSS_(cal-uncal)_median": rss_del_ci[1],
            "Delta_RSS_(cal-uncal)_ci97p5": rss_del_ci[2],

            "reps": int(sub.shape[0]),
            "plots_used_median": float(np.median(sub["plots_used"].to_numpy(float))),
            "n_test_total_median": float(np.median(sub["n_test_total"].to_numpy(float))),
        })

    return pd.DataFrame(out).sort_values(["Model", "n_cal"])


# =========================================================
# LOPO helpers
# =========================================================
def _robust_p0_map_excluding_plot(exclude_plot: str) -> dict:
    """
    Costruisce robust_p0 per ogni modello usando by_plot_df
    escludendo il plot hold-out (no leakage).
    """
    out_map = {}
    for mname in model_order:
        spec = MODEL_SPECS[mname]
        k = spec["k"]
        lo, hi = map(np.asarray, spec["bounds"])

        sub = by_plot_df[
            (by_plot_df["Model"] == mname) &
            (by_plot_df["success"] == True) &
            (by_plot_df["Plot"].astype(str) != str(exclude_plot))
        ].copy()

        robust_p0 = None
        if not sub.empty:
            P = sub[[f"p{i+1}" for i in range(k)]].to_numpy(float)
            P = P[np.isfinite(P).all(axis=1)]
            P2 = [p for p in P if not _near_bounds(p, lo, hi, rel=0.01)]
            P2 = np.asarray(P2, float) if len(P2) else np.empty((0, k))

            if P2.shape[0] >= 2:
                robust_p0 = np.nanmedian(P2, axis=0)
            elif P.shape[0] >= 2:
                robust_p0 = np.nanmedian(P, axis=0)

        if robust_p0 is None or (not np.all(np.isfinite(robust_p0))):
            robust_p0 = np.asarray(spec["p0_fallback"], float)

        eps = 1e-6
        robust_p0 = np.minimum(np.maximum(robust_p0, lo + eps), hi - eps)
        out_map[mname] = robust_p0.tolist()

    return out_map


def lopo_sparse_eval(
    n_list: list[int],
    reps: int,
    seed: int,
    h0: float = 1.3,
    a_min: float = 0.1,
    a_max: float = 5.0,
    strict_p0: bool = True,
) -> pd.DataFrame:
    """
    LOPO:
      per ogni plot held-out:
        - fit globale su altri plot
        - pred su plot held-out
        - per ogni n_cal e rep: stima a_plot su n_cal alberi, valuta su restanti
    """
    idx_by_plot = {pl: np.where(G_all == pl)[0] for pl in plots}

    rng_master = np.random.default_rng(seed)
    fold_seeds = rng_master.integers(0, 1_000_000_000, size=len(plots), dtype=np.int64)

    rows = []

    for fold_i, pl_test in enumerate(plots):
        test_idx = idx_by_plot[pl_test]
        if test_idx.size < 3:
            continue

        train_mask = np.ones_like(G_all, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        D_tr, H_tr = D_all[train_idx], H_all[train_idx]
        D_te, H_te = D_all[test_idx], H_all[test_idx]

        p0_fold_map = _robust_p0_map_excluding_plot(pl_test) if strict_p0 else robust_p0_map
        rng = np.random.default_rng(int(fold_seeds[fold_i]))

        for mname in model_order:
            spec = MODEL_SPECS[mname]
            func = spec["func"]
            lo, hi = spec["bounds"]

            p0 = p0_fold_map.get(mname, spec["p0_fallback"])
            popt, ok, _err = _safe_curve_fit(func, D_tr, H_tr, p0=p0, bounds=(lo, hi), maxfev=60000)
            if not ok or (not np.all(np.isfinite(popt))):
                continue

            yhat_te = func(D_te, *popt)
            if not np.all(np.isfinite(yhat_te)):
                continue

            n_te = int(H_te.size)
            pos = np.arange(n_te)

            for rep in range(reps):
                for n_cal in n_list:
                    if n_te <= n_cal:
                        continue

                    cal_pos = rng.choice(pos, size=int(n_cal), replace=False)
                    test_pos = np.setdiff1d(pos, cal_pos, assume_unique=False)

                    a = _fit_a_from_subset(
                        yhat_subset=yhat_te[cal_pos],
                        h_subset=H_te[cal_pos],
                        h0=h0,
                        a_min=a_min,
                        a_max=a_max,
                    )

                    pred_cal = _predict_calibrated_with_a(yhat_te[test_pos], a=a, h0=h0)
                    pred_unc = yhat_te[test_pos]
                    obs = H_te[test_pos]

                    met_cal = metrics_plain(obs, pred_cal)
                    met_unc = metrics_plain(obs, pred_unc)

                    # ---- include Bias + RSS in long rows
                    rows.append({
                        "Model": mname,
                        "fold_plot": str(pl_test),
                        "rep": int(rep),
                        "n_cal": int(n_cal),
                        "n_test": int(test_pos.size),

                        "RMSE_test_cal": float(met_cal["RMSE"]),
                        "RMSE_test_uncal": float(met_unc["RMSE"]),
                        "Delta_RMSE_(cal-uncal)": float(met_cal["RMSE"] - met_unc["RMSE"]),

                        "Bias_test_cal": float(met_cal["Bias"]),
                        "Bias_test_uncal": float(met_unc["Bias"]),
                        "Delta_Bias_(cal-uncal)": float(met_cal["Bias"] - met_unc["Bias"]),

                        "RSS_test_cal": float(met_cal["RSS"]),
                        "RSS_test_uncal": float(met_unc["RSS"]),
                        "Delta_RSS_(cal-uncal)": float(met_cal["RSS"] - met_unc["RSS"]),
                    })

    return pd.DataFrame(rows)


def summarize_lopo(df_long: pd.DataFrame) -> pd.DataFrame:
    # ---- summary now includes RMSE + Bias + RSS (and deltas)
    if df_long.empty:
        return pd.DataFrame()

    def q(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        return (float(np.quantile(v, 0.025)),
                float(np.quantile(v, 0.50)),
                float(np.quantile(v, 0.975)))

    out = []
    for (m, n_cal), sub in df_long.groupby(["Model", "n_cal"]):
        rmse_cal = sub["RMSE_test_cal"].to_numpy(float)
        rmse_unc = sub["RMSE_test_uncal"].to_numpy(float)
        rmse_del = rmse_cal - rmse_unc

        bias_cal = sub["Bias_test_cal"].to_numpy(float)
        bias_unc = sub["Bias_test_uncal"].to_numpy(float)
        bias_del = bias_cal - bias_unc

        rss_cal = sub["RSS_test_cal"].to_numpy(float)
        rss_unc = sub["RSS_test_uncal"].to_numpy(float)
        rss_del = rss_cal - rss_unc

        rmse_cal_ci = q(rmse_cal); rmse_unc_ci = q(rmse_unc); rmse_del_ci = q(rmse_del)
        bias_cal_ci = q(bias_cal); bias_unc_ci = q(bias_unc); bias_del_ci = q(bias_del)
        rss_cal_ci  = q(rss_cal);  rss_unc_ci  = q(rss_unc);  rss_del_ci  = q(rss_del)

        out.append({
            "Model": m,
            "n_cal": int(n_cal),

            "RMSE_test_cal_ci2p5": rmse_cal_ci[0],
            "RMSE_test_cal_median": rmse_cal_ci[1],
            "RMSE_test_cal_ci97p5": rmse_cal_ci[2],
            "RMSE_test_uncal_ci2p5": rmse_unc_ci[0],
            "RMSE_test_uncal_median": rmse_unc_ci[1],
            "RMSE_test_uncal_ci97p5": rmse_unc_ci[2],
            "Delta_RMSE_(cal-uncal)_ci2p5": rmse_del_ci[0],
            "Delta_RMSE_(cal-uncal)_median": rmse_del_ci[1],
            "Delta_RMSE_(cal-uncal)_ci97p5": rmse_del_ci[2],

            "Bias_test_cal_ci2p5": bias_cal_ci[0],
            "Bias_test_cal_median": bias_cal_ci[1],
            "Bias_test_cal_ci97p5": bias_cal_ci[2],
            "Bias_test_uncal_ci2p5": bias_unc_ci[0],
            "Bias_test_uncal_median": bias_unc_ci[1],
            "Bias_test_uncal_ci97p5": bias_unc_ci[2],
            "Delta_Bias_(cal-uncal)_ci2p5": bias_del_ci[0],
            "Delta_Bias_(cal-uncal)_median": bias_del_ci[1],
            "Delta_Bias_(cal-uncal)_ci97p5": bias_del_ci[2],

            "RSS_test_cal_ci2p5": rss_cal_ci[0],
            "RSS_test_cal_median": rss_cal_ci[1],
            "RSS_test_cal_ci97p5": rss_cal_ci[2],
            "RSS_test_uncal_ci2p5": rss_unc_ci[0],
            "RSS_test_uncal_median": rss_unc_ci[1],
            "RSS_test_uncal_ci97p5": rss_unc_ci[2],
            "Delta_RSS_(cal-uncal)_ci2p5": rss_del_ci[0],
            "Delta_RSS_(cal-uncal)_median": rss_del_ci[1],
            "Delta_RSS_(cal-uncal)_ci97p5": rss_del_ci[2],

            "reps": int(sub.shape[0]),
            "folds_used": int(sub["fold_plot"].nunique()),
            "n_test_median": float(np.median(sub["n_test"].to_numpy(float))),
        })

    return pd.DataFrame(out).sort_values(["Model", "n_cal"])


# =========================================================
# BOOTSTRAP utilities (stability / robustness)
# =========================================================
def bootstrap_indices(G: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    G = np.asarray(G)
    n = G.size
    uniq = np.unique(G)

    if mode == "flat":
        return rng.integers(0, n, size=n)

    if mode == "within_plot":
        idxs = []
        for pl in uniq:
            pl_idx = np.where(G == pl)[0]
            idxs.append(rng.choice(pl_idx, size=pl_idx.size, replace=True))
        return np.concatenate(idxs)

    if mode == "cluster_plot":
        sampled_plots = rng.choice(uniq, size=uniq.size, replace=True)
        idxs = []
        for pl in sampled_plots:
            pl_idx = np.where(G == pl)[0]
            idxs.append(pl_idx)
        return np.concatenate(idxs)

    raise ValueError(f"Unknown bootstrap mode: {mode}")


def _fit_one_model_on_sample(mname: str, D: np.ndarray, H: np.ndarray, G: np.ndarray,
                            p0_map: dict, do_calibration: bool = True) -> dict:
    spec = MODEL_SPECS[mname]
    func = spec["func"]
    k = spec["k"]
    lo, hi = spec["bounds"]

    p0 = p0_map.get(mname, spec["p0_fallback"])
    popt, ok, err = _safe_curve_fit(func, D, H, p0=p0, bounds=(lo, hi), maxfev=60000)

    row = {"Model": mname, "success": bool(ok), "err": "" if ok else err}
    for i in range(k):
        row[f"p{i+1}"] = float(popt[i]) if (ok and np.isfinite(popt[i])) else np.nan

    row["n_plots_eff"] = int(len(np.unique(G)))

    if not ok:
        for kk in ["RMSE", "R2_adj", "Bias", "RSS", "AIC",
                   "RMSE_cal", "R2_adj_cal", "Bias_cal", "RSS_cal", "AIC_cal"]:
            row[kk] = np.nan
        return row

    yhat = func(D, *popt)
    met_uncal = metrics(H, yhat, k=k)
    for kk in ["RMSE", "R2_adj", "Bias", "RSS", "AIC"]:
        row[kk] = float(met_uncal[kk])

    if do_calibration:
        scales, _ = calibrate_scales_by_plot(yhat, H, G, h0=h0, a_min=0.1, a_max=5.0)
        a_vec = np.array([scales.get(g, np.nan) for g in G], dtype=float)
        yhat_cal = h0 + a_vec * (yhat - h0)

        k_eff = k + int(len(np.unique(G)))
        met_cal = metrics(H, yhat_cal, k=k_eff)
        row["RMSE_cal"] = float(met_cal["RMSE"])
        row["R2_adj_cal"] = float(met_cal["R2_adj"])
        row["Bias_cal"] = float(met_cal["Bias"])
        row["RSS_cal"] = float(met_cal["RSS"])
        row["AIC_cal"] = float(met_cal["AIC"])
    else:
        for kk in ["RMSE_cal", "R2_adj_cal", "Bias_cal", "RSS_cal", "AIC_cal"]:
            row[kk] = np.nan

    return row


def bootstrap_models(B: int = 500, mode: str = "within_plot", seed: int = 0,
                     do_calibration: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows_ = []
    for b in range(B):
        idx = bootstrap_indices(G_all, mode=mode, rng=rng)
        Db = D_all[idx]
        Hb = H_all[idx]
        Gb = G_all[idx]

        for mname in model_order:
            r = _fit_one_model_on_sample(
                mname, Db, Hb, Gb,
                p0_map=robust_p0_map,
                do_calibration=do_calibration
            )
            r["b"] = int(b)
            r["mode"] = str(mode)
            r["seed"] = int(seed)
            rows_.append(r)

    return pd.DataFrame(rows_)


def summarize_bootstrap(boot_df: pd.DataFrame, cols_to_summarize: list[str]) -> pd.DataFrame:
    out = []
    for mname, sub in boot_df.groupby("Model"):
        sub_ok = sub[sub["success"] == True].copy()
        row = {
            "Model": mname,
            "B": int(sub.shape[0]),
            "success_rate": float(np.mean(sub["success"].to_numpy(bool))),
        }
        for c in cols_to_summarize:
            x = sub_ok[c].to_numpy(float) if c in sub_ok.columns else np.array([], dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_std"] = np.nan
                row[f"{c}_median"] = np.nan
                row[f"{c}_ci2p5"] = np.nan
                row[f"{c}_ci97p5"] = np.nan
            else:
                row[f"{c}_mean"] = float(np.mean(x))
                row[f"{c}_std"] = float(np.std(x, ddof=1)) if x.size >= 2 else 0.0
                row[f"{c}_median"] = float(np.median(x))
                row[f"{c}_ci2p5"] = float(np.quantile(x, 0.025))
                row[f"{c}_ci97p5"] = float(np.quantile(x, 0.975))
        out.append(row)

    return pd.DataFrame(out).sort_values("Model")


def bootstrap_ci_table_like_figure(
    boot_df: pd.DataFrame,
    rmse_col: str = "RMSE",     # oppure "RMSE_cal"
    bias_col: str = "Bias",     # oppure "Bias_cal"
    rss_col: str = "RSS",       # oppure "RSS_cal"
    digits: int = 3,
) -> pd.DataFrame:
    """
    Tabella stile figura:
    Model | RMSE (95% CI) | Bias (95% CI) | RSS (95% CI) | P(best)
    P(best) = probabilità che il modello sia il migliore (min RMSE) nel bootstrap.
    """
    def ci_str(x: np.ndarray) -> str:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return ""
        lo = float(np.quantile(x, 0.025))
        hi = float(np.quantile(x, 0.975))
        return f"{lo:.{digits}f} \u2013 {hi:.{digits}f}"  # en-dash

    need = {"Model", "b", "success", rmse_col}
    if not need.issubset(set(boot_df.columns)):
        raise ValueError(f"boot_df must contain columns {need}")

    sub = boot_df[(boot_df["success"] == True) & np.isfinite(boot_df[rmse_col])].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Model", "RMSE", "Bias", "RSS", "P(best)"])

    piv = sub.pivot_table(index="b", columns="Model", values=rmse_col, aggfunc="first")
    winners = piv.idxmin(axis=1, skipna=True)
    pbest = winners.value_counts(normalize=True)

    out_rows = []
    for mname, g in boot_df.groupby("Model"):
        g_ok = g[g["success"] == True].copy()

        rmse_vals = g_ok[rmse_col].to_numpy(float) if rmse_col in g_ok.columns else np.array([])
        bias_vals = g_ok[bias_col].to_numpy(float) if bias_col in g_ok.columns else np.array([])
        rss_vals  = g_ok[rss_col].to_numpy(float)  if rss_col  in g_ok.columns else np.array([])

        out_rows.append({
            "Model": pretty.get(mname, mname),
            "RMSE": ci_str(rmse_vals),
            "Bias": ci_str(bias_vals),
            "RSS":  ci_str(rss_vals),
            "P(best)": float(pbest.get(mname, 0.0)),
        })

    out = pd.DataFrame(out_rows)

    order_map = {pretty.get(m, m): i for i, m in enumerate(model_order)}
    out["__ord"] = out["Model"].map(order_map).fillna(9999)
    out = out.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)

    out["P(best)"] = out["P(best)"].map(lambda v: f"{v:.3f}")
    return out


# =========================================================
# CALIBRATION RUN
# =========================================================
cal_rows = []
cal_byplot_metrics_rows = []
cal_global_metrics_rows = []

for mname in model_order:
    res = results[mname]
    if not res["ok"] or (not np.all(np.isfinite(res["yhat"]))):
        continue

    yhat = res["yhat"]
    scales, diag = calibrate_scales_by_plot(yhat, H_all, G_all, h0=h0, a_min=0.1, a_max=5.0)

    a_vec = np.array([scales.get(g, np.nan) for g in G_all], dtype=float)
    yhat_cal = h0 + a_vec * (yhat - h0)

    for pl in plots:
        dpl = diag.get(pl, {})
        cal_rows.append({
            "Model": mname,
            "Plot": pl,
            "a_plot": float(scales.get(pl, np.nan)) if np.isfinite(scales.get(pl, np.nan)) else np.nan,
            "n": int(dpl.get("n", 0)),
            "num": float(dpl.get("num", np.nan)),
            "denom": float(dpl.get("denom", np.nan)),
        })

    for pl in plots:
        mask = (G_all == pl)
        if np.sum(mask) < 3:
            continue
        y_obs = H_all[mask]
        y_pr = yhat_cal[mask]
        met = metrics(y_obs, y_pr, k=1)
        cal_byplot_metrics_rows.append({"Model": mname, "Plot": pl, **met})

    k_eff = MODEL_SPECS[mname]["k"] + n_plots
    met_g = metrics(H_all, yhat_cal, k=k_eff)
    met_plain = metrics_plain(H_all, yhat_cal)

    cal_global_metrics_rows.append({
        "Model": mname,
        "k_global": MODEL_SPECS[mname]["k"],
        "n_plots": n_plots,
        "k_eff_for_AIC_R2adj": k_eff,
        "RMSE_cal_dfadj": met_g["RMSE"],
        "R2_adj_cal": met_g["R2_adj"],
        "Bias_cal": met_g["Bias"],
        "RSS_cal": met_g["RSS"],
        "AIC_cal_kEff": met_g["AIC"],
        "RMSE_cal_plain": met_plain["RMSE"],
        "R2_cal_plain": met_plain["R2"],
    })

    # obs vs pred plot (global vs calibrated)
    fig, ax = plt.subplots(figsize=(6.8, 6.6))
    ax.scatter(H_all, yhat, s=10, alpha=0.20, label="Global pred", color="0.4")
    ax.scatter(H_all, yhat_cal, s=10, alpha=0.20, label="Calibrated pred", color=COLORS[mname])
    lo_ = float(np.nanmin(H_all)); hi_ = float(np.nanmax(H_all))
    ax.plot([lo_, hi_], [lo_, hi_], linestyle="--", linewidth=2.0, color="0.2")
    stylize_axes(ax, xlabel="Observed height (m)", ylabel="Predicted height (m)", label_size=15, tick_size=12, spine_lw=1.5)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(CAL_DIR, f"obs_vs_pred_{mname}.png"), dpi=160)
    plt.close(fig)

# Save calibration outputs
CAL_SCALE_CSV = "hd_calibration_scale_by_plot_M1_to_M5.csv"
CAL_GLOBAL_CSV = "hd_metrics_global_calibrated_M1_to_M5.csv"
CAL_BY_PLOT_METRICS_CSV = "hd_metrics_by_plot_calibrated_M1_to_M5.csv"

pd.DataFrame(cal_rows).to_csv(CAL_SCALE_CSV, index=False)
pd.DataFrame(cal_global_metrics_rows).to_csv(CAL_GLOBAL_CSV, index=False)
pd.DataFrame(cal_byplot_metrics_rows).to_csv(CAL_BY_PLOT_METRICS_CSV, index=False)


# =========================================================
# SPARSE CALIBRATION EVALUATION RUN (hold-out within plot)
# =========================================================
if DO_SPARSE_CAL_EVAL:
    all_long = []
    for mname in model_order:
        df_long = sparse_calibration_eval_for_model(
            mname=mname,
            n_list=SPARSE_CAL_N_LIST,
            reps=SPARSE_CAL_REPS,
            seed=SPARSE_CAL_SEED,
            h0=h0,
            a_min=SPARSE_A_MIN,
            a_max=SPARSE_A_MAX,
        )
        if not df_long.empty:
            all_long.append(df_long)

    if all_long:
        sparse_long = pd.concat(all_long, ignore_index=True)
        sparse_sum = summarize_sparse_eval(sparse_long)

        long_csv = os.path.join(SPARSE_DIR, "hd_sparsecal_long_M1_to_M5.csv")
        sum_csv = os.path.join(SPARSE_DIR, "hd_sparsecal_summary_M1_to_M5.csv")
        sparse_long.to_csv(long_csv, index=False)
        sparse_sum.to_csv(sum_csv, index=False)

        print("\nSPARSE CALIBRATION (hold-out within plot) — SUMMARY:")

        # ---- print now includes Bias + RSS (+ deltas)
        sparse_sum_print = pd.DataFrame({
            "Model": sparse_sum["Model"].map(pretty).fillna(sparse_sum["Model"]),
            "n_cal": sparse_sum["n_cal"].astype(int),

            "RMSE_cal": np.round(sparse_sum["RMSE_test_cal_median"].to_numpy(float), 3),
            "RMSE_uncal": np.round(sparse_sum["RMSE_test_uncal_median"].to_numpy(float), 3),
            "ΔRMSE": np.round(sparse_sum["Delta_RMSE_(cal-uncal)_median"].to_numpy(float), 3),

            "Bias_cal": np.round(sparse_sum["Bias_test_cal_median"].to_numpy(float), 3),
            "Bias_uncal": np.round(sparse_sum["Bias_test_uncal_median"].to_numpy(float), 3),
            "ΔBias": np.round(sparse_sum["Delta_Bias_(cal-uncal)_median"].to_numpy(float), 3),

            "RSS_cal": np.round(sparse_sum["RSS_test_cal_median"].to_numpy(float), 3),
            "RSS_uncal": np.round(sparse_sum["RSS_test_uncal_median"].to_numpy(float), 3),
            "ΔRSS": np.round(sparse_sum["Delta_RSS_(cal-uncal)_median"].to_numpy(float), 3),

            "reps": sparse_sum["reps"].astype(int),
        }).sort_values(["Model", "n_cal"])

        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(sparse_sum_print.to_string(index=False))

        # Figure: RMSE median vs n_cal
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        for mname in model_order:
            sub = sparse_sum[sparse_sum["Model"] == mname].copy()
            if sub.empty:
                continue
            xs = sub["n_cal"].to_numpy(int)
            ys = sub["RMSE_test_cal_median"].to_numpy(float)
            ylo = sub["RMSE_test_cal_ci2p5"].to_numpy(float)
            yhi = sub["RMSE_test_cal_ci97p5"].to_numpy(float)

            ax.plot(xs, ys, marker="o", linewidth=2.0, label=pretty.get(mname, mname))
            ax.fill_between(xs, ylo, yhi, alpha=0.15)

        stylize_axes(ax, xlabel="# trees (n_cal)",
                     ylabel="RMSE (test hold-out)", label_size=14, tick_size=12, spine_lw=1.5)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        fig_path = os.path.join(SPARSE_DIR, "hd_sparsecal_rmse_vs_ncal.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    else:
        print("\nSPARSE CALIBRATION (hold-out within plot) — SUMMARY:")
        print("No sparse-calibration results (global fits failed or insufficient data).")


# =========================================================
# LOPO RUN (Leave-One-Plot-Out)
# =========================================================
if DO_LOPO_EVAL:
    lopo_long = lopo_sparse_eval(
        n_list=LOPO_N_LIST,
        reps=int(LOPO_REPS),
        seed=int(LOPO_SEED),
        h0=h0,
        a_min=LOPO_A_MIN,
        a_max=LOPO_A_MAX,
        strict_p0=bool(LOPO_STRICT_P0),
    )

    if not lopo_long.empty:
        lopo_sum = summarize_lopo(lopo_long)

        long_csv = os.path.join(LOPO_DIR, "hd_lopo_long_M1_to_M5.csv")
        sum_csv = os.path.join(LOPO_DIR, "hd_lopo_summary_M1_to_M5.csv")
        lopo_long.to_csv(long_csv, index=False)
        lopo_sum.to_csv(sum_csv, index=False)

        print("\nLOPO (leave-one-plot-out) — SUMMARY:")

        # ---- print now includes Bias + RSS (+ deltas)
        lopo_sum_print = pd.DataFrame({
            "Model": lopo_sum["Model"].map(pretty).fillna(lopo_sum["Model"]),
            "n_cal": lopo_sum["n_cal"].astype(int),

            "RMSE_cal": np.round(lopo_sum["RMSE_test_cal_median"].to_numpy(float), 3),
            "RMSE_uncal": np.round(lopo_sum["RMSE_test_uncal_median"].to_numpy(float), 3),
            "ΔRMSE": np.round(lopo_sum["Delta_RMSE_(cal-uncal)_median"].to_numpy(float), 3),

            "Bias_cal": np.round(lopo_sum["Bias_test_cal_median"].to_numpy(float), 3),
            "Bias_uncal": np.round(lopo_sum["Bias_test_uncal_median"].to_numpy(float), 3),
            "ΔBias": np.round(lopo_sum["Delta_Bias_(cal-uncal)_median"].to_numpy(float), 3),

            "RSS_cal": np.round(lopo_sum["RSS_test_cal_median"].to_numpy(float), 3),
            "RSS_uncal": np.round(lopo_sum["RSS_test_uncal_median"].to_numpy(float), 3),
            "ΔRSS": np.round(lopo_sum["Delta_RSS_(cal-uncal)_median"].to_numpy(float), 3),

            "folds": lopo_sum["folds_used"].astype(int),
            "reps": lopo_sum["reps"].astype(int),
        }).sort_values(["Model", "n_cal"])

        with pd.option_context("display.max_columns", None, "display.width", 240):
            print(lopo_sum_print.to_string(index=False))

        # Figure: RMSE (median) vs n_cal con banda CI
        fig, ax = plt.subplots(figsize=(10.0, 6.0))
        for mname in model_order:
            sub = lopo_sum[lopo_sum["Model"] == mname].copy()
            if sub.empty:
                continue
            xs = sub["n_cal"].to_numpy(int)
            ys = sub["RMSE_test_cal_median"].to_numpy(float)
            ylo = sub["RMSE_test_cal_ci2p5"].to_numpy(float)
            yhi = sub["RMSE_test_cal_ci97p5"].to_numpy(float)

            ax.plot(xs, ys, marker="o", linewidth=2.0, label=pretty.get(mname, mname))
            ax.fill_between(xs, ylo, yhi, alpha=0.15)

        stylize_axes(ax, xlabel="# trees (n_cal)",
                     ylabel="RMSE (LOPO test)", label_size=14, tick_size=12, spine_lw=1.5)
        ax.legend(frameon=False, fontsize=10)
        fig.tight_layout()
        fig_path = os.path.join(LOPO_DIR, "hd_lopo_rmse_vs_ncal.png")
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
    else:
        print("\nLOPO (leave-one-plot-out) — SUMMARY:")
        print("No LOPO results (insufficient data or fits failed).")


# =========================================================
# FIGURA SOLO M2: globale (continua) vs calibrata (tratteggiata)
# (la linea calibrata usa a_plot tipico = mediana degli a_plot)
# =========================================================
mname = "M2_Logistic"
if mname in results and results[mname]["ok"] and np.all(np.isfinite(results[mname]["yhat"])):
    res = results[mname]

    yhat_all = res["yhat"]
    scales_M2, _ = calibrate_scales_by_plot(yhat_all, H_all, G_all, h0=h0, a_min=0.1, a_max=5.0)

    a_vals = np.array([a for a in scales_M2.values() if np.isfinite(a)], dtype=float)
    a_typ = float(np.median(a_vals)) if a_vals.size else 1.0

    diam_grid = np.linspace(D_all.min(), D_all.max(), 400)

    y_global = res["predict"](diam_grid)
    y_cal = h0 + a_typ * (y_global - h0)

    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    ax.scatter(D_all, H_all, alpha=0.20, s=12, color="0.6", label="Observed")

    ax.plot(diam_grid, y_global,
            color=COLORS[mname], linewidth=3.2, linestyle="-",
            label="M2 Global (before calibration)")

    ax.plot(diam_grid, y_cal,
            color=COLORS[mname], linewidth=3.2, linestyle="--",
            label=f"M2 Calibrated (median a_plot={a_typ:.2f})")

    stylize_axes(ax, xlabel="Diameter (cm)", ylabel="Height (m)",
                 label_size=18, tick_size=14, spine_lw=1.7)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    out_path = os.path.join(CAL_DIR, "hd_M2_global_vs_calibrated.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# PRINT: ONLY METRICS (WITH calibration)
if len(cal_global_metrics_rows) == 0:
    print("\nGLOBAL METRICS (WITH calibration):")
    print("No calibrated metrics were produced (no valid models).")
else:
    cal_global_df = pd.DataFrame(cal_global_metrics_rows)
    metrics_cal = cal_global_df[[
        "Model",
        "RMSE_cal_dfadj",
        "R2_adj_cal",
        "Bias_cal",
        "RSS_cal",
        "AIC_cal_kEff"
    ]].copy()

    metrics_cal = metrics_cal.rename(columns={
        "RMSE_cal_dfadj": "RMSE",
        "R2_adj_cal": "R2_adj",
        "Bias_cal": "Bias",
        "RSS_cal": "RSS",
        "AIC_cal_kEff": "AIC",
    }).sort_values("Model")

    print("\nGLOBAL METRICS (WITH calibration):")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(metrics_cal.to_string(index=False))


# -------------------------
# 6. Diagnostics (global, M1–M5)
# -------------------------
DIAG_DIR = "hd_diagnostics"
os.makedirs(DIAG_DIR, exist_ok=True)

model_curves = {}
for name in model_order:
    if not results[name]["ok"]:
        continue
    yhat = results[name]["yhat"]
    if not np.all(np.isfinite(yhat)):
        continue
    resid = H_all - yhat
    mu = float(np.nanmean(resid))
    sd = float(np.sqrt(np.nanmean((resid - mu) ** 2))) or 1.0
    z = (resid - mu) / sd
    model_curves[name] = {"fitted": yhat, "zresid": z}

if len(model_curves) > 0:
    # ---------- Q-Q (overlay) ----------
    N = len(H_all)
    ppos = (np.arange(1, N + 1) - 0.5) / N
    q_theor = stats.norm.ppf(ppos)

    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    for name in model_order:
        if name not in model_curves:
            continue
        z_sorted = np.sort(model_curves[name]["zresid"])
        ax.plot(q_theor, z_sorted, linewidth=2.4, color=COLORS[name], label=pretty[name])

    qmin, qmax = np.nanmin(q_theor), np.nanmax(q_theor)
    ax.plot([qmin, qmax], [qmin, qmax], linestyle="--", linewidth=2.0, color="0.25")

    stylize_axes(
        ax,
        xlabel="Theoretical quantiles (N(0,1))",
        ylabel="Sample quantiles (standardized)",
        label_size=15,
        tick_size=12,
        spine_lw=1.5,
    )
    ax.legend(fontsize=9, ncol=2, frameon=False)
    fig.tight_layout()
    QQ_OUT = os.path.join(DIAG_DIR, "qq_residuals_M1_to_M5.png")
    fig.savefig(QQ_OUT, dpi=160)
    plt.close(fig)

    # ---------- Residuals vs Fitted: OVERLAY (single axes) ----------
    all_fitted = np.concatenate([model_curves[n]["fitted"] for n in model_order if n in model_curves])
    all_z = np.concatenate([model_curves[n]["zresid"] for n in model_order if n in model_curves])
    xlo, xhi = float(np.nanmin(all_fitted)), float(np.nanmax(all_fitted))
    ylo, yhi = float(np.nanmin(all_z)), float(np.nanmax(all_z))
    pad_x = 0.03 * (xhi - xlo + 1e-9)
    pad_y = 0.05 * (yhi - ylo + 1e-9)

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.axhline(0.0, linestyle="--", linewidth=2.0, color="0.25")

    for name in model_order:
        if name not in model_curves:
            continue
        c = COLORS[name]
        fitted = model_curves[name]["fitted"]
        z = model_curves[name]["zresid"]
        ax.scatter(fitted, z, s=12, alpha=0.16, color=c, label=pretty[name])

        if lowess is not None:
            sm = lowess(z, fitted, frac=0.25, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], linewidth=2.2, color=c)

    ax.set_xlim(xlo - pad_x, xhi + pad_x)
    ax.set_ylim(ylo - pad_y, yhi + pad_y)

    stylize_axes(
        ax,
        xlabel="Fitted height (m)",
        ylabel="Standardized residuals",
        label_size=15,
        tick_size=12,
        spine_lw=1.5,
    )
    ax.legend(fontsize=9, ncol=2, frameon=False)

    fig.tight_layout()
    out_path = os.path.join(DIAG_DIR, "resid_vs_fitted_M1_to_M5.png")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# =========================================================
# BOOTSTRAP RUN (after fits + calibration + diagnostics)
# =========================================================
if DO_BOOTSTRAP:
    boot_df = bootstrap_models(
        B=int(BOOT_B),
        mode=str(BOOT_MODE),
        seed=int(BOOT_SEED),
        do_calibration=bool(BOOT_DO_CAL)
    )

    BOOT_LONG_CSV = f"hd_bootstrap_long_M1_to_M5_{BOOT_MODE}_B{BOOT_B}.csv"
    boot_df.to_csv(BOOT_LONG_CSV, index=False)

    # =========================================================
    # EXTRA FIGURE: histogram of ΔRMSE_cal = M2 - M1 (NO TITLE)
    # =========================================================
    def _plot_delta_rmse_cal_hist_no_title(
        boot_df: pd.DataFrame,
        m2: str = "M2_Logistic",
        m1: str = "M1_ChapmanRichards",
        out_dir: str = SPARSE_DIR,
        out_name: str = "bootstrap_delta_RMSE_cal_M2_minus_M1.png",
        bins: int = 30,
    ) -> None:
        needed_cols = {"Model", "b", "RMSE_cal", "success"}
        if not needed_cols.issubset(set(boot_df.columns)):
            return

        sub = boot_df[(boot_df["success"] == True) & np.isfinite(boot_df["RMSE_cal"])].copy()

        m2_df = sub[sub["Model"] == m2][["b", "RMSE_cal"]].rename(columns={"RMSE_cal": "RMSE_cal_M2"})
        m1_df = sub[sub["Model"] == m1][["b", "RMSE_cal"]].rename(columns={"RMSE_cal": "RMSE_cal_M1"})

        merged = pd.merge(m2_df, m1_df, on="b", how="inner")
        if merged.empty:
            return

        delta = (merged["RMSE_cal_M2"] - merged["RMSE_cal_M1"]).to_numpy(float)
        delta = delta[np.isfinite(delta)]
        if delta.size == 0:
            return

        med = float(np.median(delta))

        os.makedirs(out_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        ax.hist(delta, bins=bins, edgecolor="0.35", linewidth=1.0, alpha=0.75)

        ax.axvline(0.0, color="0.15", linewidth=2.0)          # zero line
        ax.axvline(med, color="0.15", linewidth=2.0, linestyle="--")

        stylize_axes(ax, xlabel="ΔRMSE_cal  (M2 − M1)", ylabel="Count",
                     label_size=16, tick_size=12, spine_lw=1.5)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, out_name), dpi=200)
        plt.close(fig)

    _plot_delta_rmse_cal_hist_no_title(
        boot_df=boot_df,
        m2="M2_Logistic",
        m1="M1_ChapmanRichards",
        out_dir=SPARSE_DIR,
        out_name="bootstrap_delta_RMSE_cal_M2_minus_M1.png",
        bins=30,
    )

    max_k = int(max(MODEL_SPECS[m]["k"] for m in model_order))
    param_cols = [f"p{i+1}" for i in range(max_k)]
    metric_cols = ["RMSE", "R2_adj", "Bias", "RSS", "AIC"]
    metric_cols_cal = ["RMSE_cal", "R2_adj_cal", "Bias_cal", "RSS_cal", "AIC_cal"]

    cols_to_summarize = param_cols + metric_cols + metric_cols_cal
    boot_summary = summarize_bootstrap(boot_df, cols_to_summarize=cols_to_summarize)

    BOOT_SUM_CSV = f"hd_bootstrap_summary_M1_to_M5_{BOOT_MODE}_B{BOOT_B}.csv"
    boot_summary.to_csv(BOOT_SUM_CSV, index=False)

    rmse_col = "RMSE_cal" if BOOT_DO_CAL else "RMSE"
    bias_col = "Bias_cal" if BOOT_DO_CAL else "Bias"
    rss_col  = "RSS_cal"  if BOOT_DO_CAL else "RSS"

    boot_table = bootstrap_ci_table_like_figure(
        boot_df,
        rmse_col=rmse_col,
        bias_col=bias_col,
        rss_col=rss_col,
        digits=3
    )

    print("\nBOOTSTRAP (95% CI) — SUMMARY:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(boot_table.to_string(index=False))

    BOOT_TABLE_CSV = f"hd_bootstrap_table_ci95_M1_to_M5_{BOOT_MODE}_B{BOOT_B}.csv"
    boot_table.to_csv(BOOT_TABLE_CSV, index=False)
