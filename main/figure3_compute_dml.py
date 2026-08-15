#!/usr/bin/env python3
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]

TMP_DIR = BASE / "results/figure2/gimms/preseason_cache"
FP_TP = TMP_DIR / "forest_preseason_mean_eos_9m.npz"
FP_META = TMP_DIR / "forest_preseason_meta.csv"
FP_RAD = TMP_DIR / "all_pixels_rad_mean_eos_9m.npz"
FP_SOS = TMP_DIR / "all_pixels_sos.npz"
FP_LENGTHS = TMP_DIR / "all_pixels_fig2_preseason_lengths.csv"
VEG_FP = BASE / "data/veg_class_data/tables/veg_class.csv"

OUT_DIR = BASE / "results/figure3/data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP_DML = OUT_DIR / "all_pixels_dml_TPRSOS.csv"
FP_DML_SUM = OUT_DIR / "dml_TPRSOS_region_summary.csv"

P_THRESH_M = 1.0
T_SPLIT_C = 7.25
MIN_YEARS = 20
DML_CV = 2
DML_SEED = 0
SD_EPS = 1e-8
MONTH_DAYS = {m: 30 * m for m in range(1, 10)}
PRED = ["T", "P", "R", "SOS"]
B_NAMES = [f"b_{k}" for k in PRED]
ATE_NAMES = [f"ate_{k}" for k in PRED]
SE_NAMES = [f"se_{k}" for k in PRED]
P_NAMES = [f"p_{k}" for k in PRED]
REGION_ORDER = ["cold-dry", "hot-dry", "wet"]
def classify_region(t_c, p_m):
    if not (np.isfinite(t_c) and np.isfinite(p_m)):
        return None
    if p_m >= P_THRESH_M:
        return "wet"
    return "hot-dry" if t_c >= T_SPLIT_C else "cold-dry"

def standardize(y):
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    mask = np.isfinite(y)
    if mask.sum() < 5:
        return out
    sd = np.std(y[mask], ddof=0)
    if sd < SD_EPS:
        out[mask] = 0.0
        return out
    out[mask] = (y[mask] - np.mean(y[mask])) / sd
    return out

def _as_1d(a):
    return np.atleast_1d(np.asarray(a, dtype=float)).ravel()

def dml_ate(eos, t, p, rad, sos, years, min_years=MIN_YEARS):
    eos_z = standardize(eos)
    t_z = standardize(t)
    p_z = standardize(p)
    r_z = standardize(rad)
    s_z = standardize(sos)
    X_all = np.column_stack([t_z, p_z, r_z, s_z])
    k = 4
    y = eos_z
    years = np.asarray(years, dtype=float)
    mask = np.all(np.isfinite(X_all), axis=1) & np.isfinite(y) & np.isfinite(years)
    if mask.sum() < min_years:
        return None
    X_all, y = X_all[mask], y[mask]
    n = int(mask.sum())
    active = np.std(X_all, axis=0, ddof=0) >= SD_EPS
    if not np.any(active):
        return None
    X = X_all[:, active]
    n_splits = min(DML_CV, max(2, n // 5))
    if n < n_splits * 2:
        return None

    ate_full = np.zeros(k, dtype=float)
    se_full = np.full(k, np.nan)
    p_full = np.ones(k, dtype=float)
    try:
        est = LinearDML(
            model_y=LinearRegression(),
            model_t=LinearRegression(),
            discrete_treatment=False,
            cv=n_splits,
            random_state=DML_SEED,
        )
        est.fit(y, X)
        ate = _as_1d(est.const_marginal_ate())
        inf = est.const_marginal_ate_inference()
        pvals = _as_1d(inf.pvalue())
        ses = _as_1d(inf.stderr_mean)
        for j, aj in enumerate(np.where(active)[0]):
            ate_full[aj] = float(ate[j]) if j < ate.size else 0.0
            se_full[aj] = float(ses[j]) if j < ses.size else np.nan
            p_full[aj] = float(pvals[j]) if j < pvals.size else 1.0
        ate_full[~active] = 0.0
    except Exception:
        return None

    _lr = LinearRegression().fit(X, y)
    out = {n: float(c) for n, c in zip(B_NAMES, ate_full)}
    out.update({n: float(c) for n, c in zip(ATE_NAMES, ate_full)})
    out.update({n: float(s) for n, s in zip(SE_NAMES, se_full)})
    out.update({n: float(pv) for n, pv in zip(P_NAMES, p_full)})
    out["intercept"] = float(_lr.intercept_)
    out["r2"] = float(r2_score(y, _lr.predict(X)))
    out["n_years"] = n
    out["estimator"] = "LinearDML"
    out["n_active_treatments"] = int(active.sum())
    return out

def main():
    t0 = time.time()
    for fp in [FP_TP, FP_META, FP_RAD, FP_SOS, FP_LENGTHS]:
        if not fp.exists():
            raise FileNotFoundError(fp)

    print("Loading preseason cache...")
    z = np.load(FP_TP)
    zr = np.load(FP_RAD)
    zs = np.load(FP_SOS)
    meta = pd.read_csv(FP_META)
    leng = pd.read_csv(FP_LENGTHS)
    years = np.asarray(z["years"], dtype=float)
    if years.size == 0:
        years = np.arange(1982, 2023, dtype=float)
    eos_mat = np.asarray(z["eos"], dtype=float)
    sos_mat = np.asarray(zs["sos"], dtype=float)
    pre_t = {m: np.asarray(z[f"t_{m}m"], dtype=float) for m in MONTH_DAYS}
    pre_p = {m: np.asarray(z[f"p_{m}m"], dtype=float) for m in MONTH_DAYS}
    pre_r = {m: np.asarray(zr[f"r_{m}m"], dtype=float) for m in MONTH_DAYS}

    df = meta.reset_index(drop=True).copy()
    df = df.merge(leng, on=["latitude", "longitude"], how="left", suffixes=("", "_y"))
    for c in ["preseason_t_months", "preseason_p_months", "preseason_r_months"]:
        if f"{c}_y" in df.columns:
            df[c] = df[c].fillna(df[f"{c}_y"])
            df.drop(columns=[f"{c}_y"], inplace=True, errors="ignore")
    if VEG_FP.exists():
        veg = pd.read_csv(VEG_FP)
        if "veg_class" not in df.columns:
            df = df.merge(
                veg[["latitude", "longitude", "veg_class"]],
                on=["latitude", "longitude"],
                how="left",
            )
    df["region"] = [classify_region(t, p) for t, p in zip(df["annual_t"], df["annual_p"])]
    # drop EN (class 11)
    if "veg_class" in df.columns:
        df_fit = df[df["veg_class"].isin([12.0, 13.0, 14.0])].copy()
    else:
        df_fit = df.copy()
    df_fit = df_fit[df_fit["region"].isin(REGION_ORDER)].reset_index(drop=True)
    meta_key = {
        (round(float(a), 5), round(float(b), 5)): i
        for i, (a, b) in enumerate(zip(meta["latitude"], meta["longitude"]))
    }
    print(
        f"Pixels to fit: {len(df_fit)} ; years={int(years[0])}–{int(years[-1])}"
    )

    rows = []
    n = len(df_fit)
    for i in range(n):
        if i and i % 5000 == 0:
            print(
                f"  dml {i}/{n} kept={len(rows)} ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        row = df_fit.iloc[i]
        lt, lp, lr = row["preseason_t_months"], row["preseason_p_months"], row["preseason_r_months"]
        if not (np.isfinite(lt) and np.isfinite(lp) and np.isfinite(lr)):
            continue
        k = (round(float(row["latitude"]), 5), round(float(row["longitude"]), 5))
        j = meta_key.get(k)
        if j is None:
            continue
        res = dml_ate(
            eos_mat[j],
            pre_t[int(lt)][j],
            pre_p[int(lp)][j],
            pre_r[int(lr)][j],
            sos_mat[j],
            years,
        )
        if res is None:
            continue
        rows.append(
            {
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "annual_t": float(row["annual_t"]),
                "annual_p": float(row["annual_p"]),
                "region": row["region"],
                "veg_class": float(row["veg_class"]) if pd.notna(row.get("veg_class", np.nan)) else np.nan,
                "preseason_t_months": int(lt),
                "preseason_p_months": int(lp),
                "preseason_r_months": int(lr),
                **res,
            }
        )

    coef = pd.DataFrame(rows)
    coef.to_csv(FP_DML, index=False)
    print(f"\nSaved {FP_DML} n={len(coef)}")
    summary = coef.groupby("region")[B_NAMES + ["r2", "n_years"]].mean().reindex(REGION_ORDER).round(3)
    summary.to_csv(FP_DML_SUM)
    print(summary.to_string())
    print("Overall means:", coef[B_NAMES].mean().round(3).to_dict())
    print(f"Done in {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
