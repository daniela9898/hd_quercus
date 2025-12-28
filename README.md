# 🌳📏 hd_quercus — Height–Diameter Modelling for *Quercus robur*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#)
[![SciPy](https://img.shields.io/badge/SciPy-curve_fit-orange.svg)](#)
[![Pandas](https://img.shields.io/badge/Pandas-data%20frames-brightgreen.svg)](#)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-figures-lightgrey.svg)](#)
[![Status](https://img.shields.io/badge/Status-Research%20code-purple.svg)](#)

A compact, reproducible pipeline for **Height–Diameter (H–D) curve modelling** in *Quercus robur*, with:

- ✅ **Per-plot fits** (variability + robust initialization)
- ✅ **Global fits** (pooled data)
- ✅ **Post-hoc per-plot calibration** (single scale factor `a_plot`)
- ✅ **Operational sparse calibration evaluation** (using only *k* measured trees per plot)
- ✅ **LOPO generalisation** (Leave-One-Plot-Out)
- ✅ **Bootstrap stability / robustness** (multiple resampling modes)

---

## ✨ What’s inside

### Models (M1–M5)

All models use **breast height offset** `h0 = 1.3 m`:

- **M1 Chapman–Richards**  
  \[
  h = 1.3 + \beta_0(1-e^{-\beta_1 d})^{\beta_2}
  \]
- **M2 Logistic**  
  \[
  h = 1.3 + \frac{\beta_0}{1 + \beta_1 e^{-\beta_2 d}}
  \]
- **M3 Näslund**  
  \[
  h = 1.3 + \frac{d^2}{(\beta_0 + \beta_1 d)^2}
  \]
- **M4 Weibull**  
  \[
  h = 1.3 + \beta_0(1-e^{-\beta_1 d^{\beta_2}})
  \]
- **M5 Wykoff**  
  \[
  h = 1.3 + \exp(\beta_0 + \beta_1/(d+1))
  \]

---

## 🧠 Method overview (the “story”)

### Step (1) — Per-plot fitting
Each model is fitted **separately per plot** to:
- capture between-plot variability
- avoid fragile global initialization
- produce **robust initial guesses** `p0` via median across plots

**Outputs**
- `hd_params_by_plot_M1_to_M5.csv`
- `hd_robust_p0_from_plots_M1_to_M5.csv`

### Step (2) — Global fitting
Using robust `p0`, fit each model to the **full dataset**.

**Outputs**
- `hd_global_M1_to_M5.png`
- `hd_params_M1_to_M5.csv`
- diagnostics in `hd_diagnostics/`

### Calibration — Plot-level localisation
After global fit, calibrate each plot with a single scale factor:

\[
h_{\text{cal}}(d) = h_0 + a_{\text{plot}}(h_{\text{global}}(d)-h_0)
\]

`a_plot` is estimated by least squares inside each plot.

**Outputs**
- `hd_calibration_scale_by_plot_M1_to_M5.csv`
- `hd_metrics_global_calibrated_M1_to_M5.csv`
- `hd_metrics_by_plot_calibrated_M1_to_M5.csv`
- `hd_calibration/obs_vs_pred_<MODELKEY>.png`
- special figure for M2:  
  `hd_calibration/hd_M2_global_vs_calibrated.png`

### Sparse calibration evaluation (operational scenario)
For each plot, estimate `a_plot` using **only `n_cal ∈ {1,3,5}` trees**, then evaluate on the remaining trees. Repeated many times.

**Outputs**
- `hd_sparse_calibration_eval/hd_sparsecal_long_M1_to_M5.csv`
- `hd_sparse_calibration_eval/hd_sparsecal_summary_M1_to_M5.csv`
- `hd_sparse_calibration_eval/hd_sparsecal_rmse_vs_ncal.png`

### LOPO evaluation (generalisation to unseen plots)
Hold out one plot at a time, fit global on the rest, then evaluate on held-out plot. Optional sparse calibration inside the held-out plot.

**Outputs**
- `hd_lopo_eval/hd_lopo_long_M1_to_M5.csv`
- `hd_lopo_eval/hd_lopo_summary_M1_to_M5.csv`
- `hd_lopo_eval/hd_lopo_rmse_vs_ncal.png`

### Bootstrap (stability / robustness)
Resampling strategies:
- `within_plot` (recommended; stratified within plots)
- `flat`
- `cluster_plot`

**Outputs**
- `hd_bootstrap_long_M1_to_M5_<MODE>_B<B>.csv`
- `hd_bootstrap_summary_M1_to_M5_<MODE>_B<B>.csv`
- `hd_bootstrap_table_ci95_M1_to_M5_<MODE>_B<B>.csv`
- extra histogram:
  `hd_sparse_calibration_eval/bootstrap_delta_RMSE_cal_M2_minus_M1.png`

---

## 📦 Requirements

- Python **3.9+** recommended
- `numpy`, `pandas`, `matplotlib`
- `scipy`
- (optional) `statsmodels` for LOWESS smooth in residual plots

Example install:
```bash
pip install numpy pandas matplotlib scipy statsmodels openpyxl
