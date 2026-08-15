#!/usr/bin/env python3
                                                                                      

import glob

import os

import time

import warnings

from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from joblib import Parallel, delayed

from scipy.optimize import dual_annealing

from scipy.spatial import cKDTree

from scipy.stats import linregress


warnings.filterwarnings("ignore", category=RuntimeWarning)


PROJ = Path("/Users/xingyihuang/Jupyter Code/Phenology/Appeal/model/projection")

MODEL = PROJ.parent

ROOT = MODEL.parent

EVAL_TMP = ROOT / "results" / "model" / "evaluation" / "gimms"

EVAL_DATA = EVAL_TMP / "data"

TABLES = MODEL / "data" / "tables" / "daily_t_p_2025-2100"

RIDGE_FP = ROOT / "results" / "figure3" / "data" / "all_pixels_ridge_TxP.csv"


OUT_DIR = ROOT / "results" / "model" / "projection" / "eval6"

OUT_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(PROJ)


CACHE_EXTRACT = EVAL_DATA / "extract_coarse_from_fine41_clim_1982_2022.npz"

if not CACHE_EXTRACT.exists():

    CACHE_EXTRACT = EVAL_DATA / "extract_200_agg_1982_2022.npz"


CACHE_L = EVAL_DATA / "siamg_preseason_p_coarse_from_fig2_L.csv"

if not CACHE_L.exists():

    CACHE_L = EVAL_DATA / "pixel_preseason_length_200_agg_1982_2022.csv"


FORCE_RERUN = True                                                                     

SSP = "ssp585"

MAXITER = 100                                           

N_JOBS = -1                                       

SEED = 42

CHILL_START_DOY = 173

F_HI_CDDP = 2500.0

F_HI_SIAMP = 3000.0

BP_MAX_CDDP = 10.0

BP_MAX_SIAMP = 10.0

CP_MAX_CDDP = 5.0

CP_MAX_SIAMP = 5.0

B_MAX_SIAM = 30.0

B_MAX_SIAMP = 30.0

START_YEAR, END_YEAR = 2025, 2100                                         

YEARS_FUT = np.arange(START_YEAR, END_YEAR + 1)

EARLY = YEARS_FUT <= 2034

LATE = YEARS_FUT >= 2090


MODEL_ORDER = ["CDD", "CDDP", "SIAM", "SIAMP"]

MODEL_COLORS = {

    "CDD": "#a3d4e1",

    "CDDP": "#5cadd8",

    "SIAM": "#ea9e87",

    "SIAMP": "#cf847e",

}


def plot_eos_trend(all_predictions, start_year=None, end_year=None, ylim=(265, 345), inset_ylim=None):

    years = np.asarray(all_predictions[0]["years"], int)

    if start_year is None:

        start_year = int(years.min())

    if end_year is None:

        end_year = int(years.max()) + 1

    df_preds = pd.DataFrame({

        "latitude": [p["latitude"] for p in all_predictions],

        "longitude": [p["longitude"] for p in all_predictions],

        "model": [p["model"] for p in all_predictions],

        "eos": [p["predicted_eos"] for p in all_predictions],

    })

    models_order = [m for m in MODEL_ORDER if m in set(df_preds["model"])]

    fig, ax_main = plt.subplots(figsize=(4.2, 4.5))

    for model in models_order:

        model_rows = df_preds[df_preds["model"] == model]

        eos_matrix = np.stack(model_rows["eos"].values)

        eos_df_model = pd.DataFrame(eos_matrix, columns=years)

        eos_mean = eos_df_model.mean(axis=0)

        eos_std = eos_df_model.std(axis=0)

        mean_smooth = eos_mean.rolling(window=1, center=True, min_periods=1).mean()

        std_smooth = eos_std.rolling(window=1, center=True, min_periods=1).mean()

        color = MODEL_COLORS.get(model, "black")

        ax_main.plot(years, mean_smooth, label=model, color=color, linewidth=2)

        ax_main.fill_between(

            years,

            mean_smooth - std_smooth / 4,

            mean_smooth + std_smooth / 4,

            color="lightgray",

            alpha=0.2,

        )

    ax_main.set_xlabel("Year", fontsize=16)

    ax_main.set_ylabel("EOS (DOY)", fontsize=16)

    ax_main.set_xlim(int(years.min()), int(years.max()))

    ax_main.tick_params(axis="both", labelsize=14)

    if ylim is not None:

        ax_main.set_ylim(*ylim)

    ylo, yhi = ax_main.get_ylim()

    yticks = np.arange(np.ceil(ylo / 20.0) * 20.0, np.floor(yhi / 20.0) * 20.0 + 0.1, 20.0)

    ax_main.set_yticks(yticks)

    ax_inset = ax_main.inset_axes([0.10, 0.58, 0.42, 0.36])

    y0, y1 = int(years.min()), int(years.max())

    diff_means, diff_errors = [], []

    for model in models_order:

        model_rows = df_preds[df_preds["model"] == model]

        eos_matrix = np.stack(model_rows["eos"].values)

        eos_df_model = pd.DataFrame(eos_matrix, columns=years)

        first_5 = eos_df_model.loc[:, y0 : y0 + 4].mean(axis=1)

        last_5 = eos_df_model.loc[:, y1 - 4 : y1].mean(axis=1)

        diff = last_5 - first_5

        diff_means.append(diff.mean())

        diff_errors.append(0.25 * diff.std())

    ax_inset.bar(

        np.arange(len(models_order)), diff_means, yerr=diff_errors, capsize=5,

        color=[MODEL_COLORS.get(m, "#cccccc") for m in models_order],

    )

    ax_inset.set_xticks(np.arange(len(models_order)))

    ax_inset.set_xticklabels(models_order, rotation=45, ha="right", fontsize=12)

    ax_inset.axhline(0, color="black", linewidth=0.5, linestyle="--")

    ax_inset.yaxis.tick_right()

    ax_inset.yaxis.set_label_position("right")

    ax_inset.set_ylabel("EOS delay (days)", fontsize=13)

    ax_inset.tick_params(axis="both", labelsize=12)

    ax_inset.tick_params(axis="y", left=False, right=True, labelleft=False, labelright=True)

    if inset_ylim is None:

        dmin = float(np.nanmin(diff_means))

        dmax = float(np.nanmax(diff_means))

        inset_ylim = (dmin - 2.0, dmax + 2.0)

    ax_inset.set_ylim(*inset_ylim)

    plt.subplots_adjust(right=0.95, bottom=0.15)

    return fig


REGION_ORDER = ["cold-dry", "hot-dry", "wet"]

REGION_TITLE = {"cold-dry": "Cold–dry", "hot-dry": "Hot–dry", "wet": "Wet"}


def day_length_vec(n_days, latitude):

    doy = np.arange(1, n_days + 1, dtype=float)

    lat = np.deg2rad(float(latitude))

    decl = 0.409 * np.sin(2.0 * np.pi * doy / 365.0 - 1.39)

    cos_omega = np.clip(-np.tan(lat) * np.tan(decl), -1.0, 1.0)

    return 24.0 * np.arccos(cos_omega) / np.pi


def _rate_matrix(Tmini, hours, T_base, P_base=None):

    temp = np.asarray(Tmini, float)

    rate = np.maximum(float(T_base) - temp, 0.0)

    if P_base is not None:

        h = np.asarray(hours, float)[:, None]

        rate = rate * np.maximum(1.0 - h / float(P_base), 0.0)

    rate[:CHILL_START_DOY, :] = 0.0

    rate[~np.isfinite(rate)] = 0.0

    return rate


def _eos_from_rate(rate, qcrit):

    qcrit = np.asarray(qcrit, float)

    if qcrit.ndim == 0:

        qcrit = np.full(rate.shape[1], float(qcrit))

    bad = ~np.isfinite(qcrit) | (qcrit <= 0)

    cum = np.cumsum(rate, axis=0)

    hit = cum >= qcrit[None, :]

    any_hit = np.any(hit, axis=0) & ~bad

    out = np.full(rate.shape[1], np.nan)

    out[any_hit] = np.argmax(hit[:, any_hit], axis=0) + 1.0

    return out


def _quad_P_threshold(F_base, Pz, b_p, c_p):

    Pz = np.asarray(Pz, float).ravel()

    q = np.asarray(F_base, float) + float(b_p) * Pz + float(c_p) * (Pz ** 2)

    return np.maximum(q, 1.0)


def CDD_model(par, data):

    Tb, F = map(float, par[:2])

    return _eos_from_rate(_rate_matrix(data["Tmini"], data["hours"], Tb), F)


def CDDP_model(par, data):

    Tb, F, b_p, c_p = map(float, par[:4])

    qcrit = _quad_P_threshold(F, data["P_z"], b_p, c_p)

    return _eos_from_rate(_rate_matrix(data["Tmini"], data["hours"], Tb), qcrit)


def SIAM_model(par, predictor, data):

    Tb, Pb, F, bs = map(float, par[:4])

    sos = np.asarray(predictor, float).ravel()

    qcrit = F + bs * sos

    return _eos_from_rate(_rate_matrix(data["Tmini"], data["hours"], Tb, Pb), qcrit)


def SIAMP_model(par, predictor, data):

    Tb, Pb, F, bs, b_p, c_p = map(float, par[:6])

    sos = np.asarray(predictor, float).ravel()

    F_base = float(F) + float(bs) * sos

    qcrit = _quad_P_threshold(F_base, data["P_z"], b_p, c_p)

    return _eos_from_rate(_rate_matrix(data["Tmini"], data["hours"], Tb, Pb), qcrit)


def get_model_list():

    return {

        "CDD": dict(fun=CDD_model, needs_sos=False, needs_tp=False,

                    lower=[0.0, 0.0], upper=[25.0, F_HI_CDDP]),

        "CDDP": dict(fun=CDDP_model, needs_sos=False, needs_tp=True,

                     lower=[0.0, 0.0, -BP_MAX_CDDP, -CP_MAX_CDDP],

                     upper=[25.0, F_HI_CDDP, BP_MAX_CDDP, CP_MAX_CDDP]),

        "SIAM": dict(fun=SIAM_model, needs_sos=True, needs_tp=False,

                     lower=[0.0, 8.0, 0.1, -B_MAX_SIAM],

                     upper=[25.0, 16.0, F_HI_SIAMP, B_MAX_SIAM]),

        "SIAMP": dict(fun=SIAMP_model, needs_sos=True, needs_tp=True,

                      lower=[0.0, 8.0, 0.1, -B_MAX_SIAMP, -BP_MAX_SIAMP, -CP_MAX_SIAMP],

                      upper=[25.0, 16.0, F_HI_SIAMP, B_MAX_SIAMP, BP_MAX_SIAMP, CP_MAX_SIAMP]),

    }


def as_doy_year(arr):

    a = np.asarray(arr, float)

    return a.T if a.shape[0] != 366 and a.shape[1] == 366 else a


def pack_data(daily_t_pix, lat, year_idx, P_z=None, T_z=None):

    Tmini = daily_t_pix[:, year_idx].copy()

    Tmini[~np.isfinite(Tmini)] = 999.0

    n_days = Tmini.shape[0]

    hours = day_length_vec(n_days, lat)

    Li = hours[:, None] * np.ones((1, len(year_idx)))

    data = {"T": Tmini, "Tmini": Tmini, "hours": hours, "Li": Li}

    if P_z is not None:

        data["P_z"] = np.asarray(P_z, float)[year_idx]

    if T_z is not None:

        data["T_z"] = np.asarray(T_z, float)[year_idx]

    return data


def predict(minfo, par, sos_vals, data):

    if minfo["needs_sos"]:

        return minfo["fun"](par, sos_vals, data)

    return minfo["fun"](par, data)


def safe_cost(par, minfo, sos_tr, dtr, obs_tr):

    if not np.all(np.isfinite(par)):

        return 1e10

    try:

        pred = predict(minfo, par, sos_tr if minfo["needs_sos"] else None, dtr)

        if pred is None or not np.all(np.isfinite(pred)):

            return 1e10

        ok = np.isfinite(pred) & np.isfinite(obs_tr)

        if ok.sum() < 3:

            return 1e10

        rmse = float(np.sqrt(np.mean((pred[ok] - obs_tr[ok]) ** 2)))

        return rmse if np.isfinite(rmse) else 1e10

    except Exception:

        return 1e10


def bias_correct_T(T_fut, T_hist):

    corr = np.nanmean(T_hist, axis=1) - np.nanmean(T_fut[:, EARLY], axis=1)

    corr[~np.isfinite(corr)] = 0.0

    return T_fut + corr[:, None]


def bias_correct_P(P_fut, P_hist):

    Ph = P_hist * 1000.0 if np.nanmean(P_hist) < 0.1 else P_hist

    corr = np.nanmean(Ph, axis=1) - np.nanmean(P_fut[:, EARLY], axis=1)

    corr[~np.isfinite(corr)] = 0.0

    return np.maximum(P_fut + corr[:, None], 0.0)


def project_sos_clipped(years_hist, sos_hist, years_fut):

    ok = np.isfinite(sos_hist)

    slope, intercept, _, _, _ = linregress(years_hist[ok], sos_hist[ok])

    pred = slope * years_fut + intercept

    lo, hi = np.nanpercentile(sos_hist[ok], [5, 95])

    return np.clip(pred, float(lo), float(hi))


def preseason_p_sum_one(daily_p_pix, mean_eos, L_star):

    if not np.isfinite(mean_eos) or not np.isfinite(L_star):

        return np.full(daily_p_pix.shape[1], np.nan)

    eos_doy = int(round(mean_eos))

    L = int(round(L_star))

    start_doy = max(1, eos_doy - L)

    end_doy = max(start_doy, eos_doy)

    sl = daily_p_pix[start_doy - 1 : end_doy, :]

    return np.nansum(sl, axis=0)


def fig3_region_from_ridge(lat, lon, ridge_df):

    sub = ridge_df[

        (np.abs(ridge_df.latitude - lat) < 1e-3)

        & (np.abs(ridge_df.longitude - lon) < 1e-3)

    ]

    if len(sub) and "region" in sub.columns and pd.notna(sub["region"].iloc[0]):

        return str(sub["region"].iloc[0])

    if lat > 55.0:

        return "cold-dry"

    if lon < -80.0:

        return "hot-dry"

    return "wet"


def load_future_var(folder, glob_pat, varname):

    files = sorted(glob.glob(str(folder / glob_pat)))

    if not files:

        raise FileNotFoundError(f"No files matched {folder / glob_pat}")

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        df["lat"] = df["lat"].round(4)

        df["lon"] = df["lon"].round(4)

        vcols = [c for c in df.columns if c not in ("lat", "lon", "year", "doy", "date")]

        col = vcols[0] if vcols else varname

        dfs.append(df[["lat", "lon", "year", "doy", col]])

        

    full_df = pd.concat(dfs, ignore_index=True)

    store = {}

    year_cols = None

    for (lat, lon), group in full_df.groupby(["lat", "lon"]):

        vcol = group.columns[-1]

        pivot = group.pivot_table(index="doy", columns="year", values=vcol).reindex(range(1, 367))

        pivot = pivot.reindex(columns=list(YEARS_FUT))

        store[(lat, lon)] = pivot.to_numpy(dtype=float)

    return store


def fit_one_pixel(r, daily_t_pi, daily_p_pi, eos, sos, preseason_p_pi, preseason_t_pi, years_hist, model_list):

    r = pd.Series(r) if not isinstance(r, pd.Series) else r

    dt = as_doy_year(daily_t_pi)

    dp = as_doy_year(daily_p_pi) if daily_p_pi is not None else np.zeros_like(dt)

    if np.nanmean(dp) < 0.1:

        dp = dp * 1000.0


    P_pre = preseason_p_pi if preseason_p_pi is not None else preseason_p_sum_one(dp, r.mean_eos, r.L_star)

    T_pre = preseason_t_pi if preseason_t_pi is not None else np.nanmean(dt[150:270, :], axis=0)


    P_mu, P_sd = np.nanmean(P_pre), np.nanstd(P_pre)

    if not np.isfinite(P_sd) or P_sd < 1e-9:

        P_sd = 1.0

    P_z_hist = (P_pre - P_mu) / P_sd


    T_mu, T_sd = np.nanmean(T_pre), np.nanstd(T_pre)

    if not np.isfinite(T_sd) or T_sd < 1e-9:

        T_sd = 1.0

    T_z_hist = (T_pre - T_mu) / T_sd


    sos_mu, sos_sd = np.nanmean(sos), np.nanstd(sos)

    if not np.isfinite(sos_sd) or sos_sd < 1e-9:

        sos_sd = 1.0

    sos_z_hist = (sos - sos_mu) / sos_sd


    valid = np.where(

        np.isfinite(eos) & np.isfinite(sos) & np.isfinite(P_pre) & np.isfinite(dt).any(axis=0)

    )[0]

    if valid.size < 10:

        return None


    params = {}

    obs_tr, sos_tr = eos[valid], sos_z_hist[valid]

    dtr = pack_data(dt, r.latitude, valid)

    dtr_tp = pack_data(dt, r.latitude, valid, P_z=P_z_hist, T_z=T_z_hist)


    for mname in MODEL_ORDER:

        minfo = model_list[mname]

        duse = dtr_tp if minfo["needs_tp"] else dtr

        res = dual_annealing(

            lambda p, _m=minfo, _d=duse: safe_cost(p, _m, sos_tr, _d, obs_tr),

            bounds=list(zip(minfo["lower"], minfo["upper"])),

            maxiter=MAXITER,

            seed=SEED,

        )

        params[mname] = np.asarray(res.x, float)


    return {

        "pixel": int(r.pixel),

        "latitude": float(r.latitude),

        "longitude": float(r.longitude),

        "region": r.region,

        "L_star": float(r.L_star),

        "mean_eos": float(r.mean_eos),

        "P_mu": float(P_mu),

        "P_sd": float(P_sd),

        "T_mu": float(T_mu),

        "T_sd": float(T_sd),

        "params": params,

        "sos_hist": np.asarray(sos, float),

        "years_hist": np.asarray(years_hist, int),

    }


def main():

    t_all = time.time()

    pred_path = OUT_DIR / f"eval6_{SSP}_predictions.pkl"

    diag_path = OUT_DIR / f"eval6_{SSP}_diagnostics.csv"


    if (not FORCE_RERUN) and pred_path.exists():

        print(f"Loading cached predictions: {pred_path}")

        all_predictions = pd.read_pickle(pred_path)

        trimmed = []

        for p in all_predictions:

            yrs = np.asarray(p["years"], int)

            eos = np.asarray(p["predicted_eos"], float)

            keep = (yrs >= START_YEAR) & (yrs <= END_YEAR)

            q = dict(p)

            q["years"] = yrs[keep]

            q["predicted_eos"] = eos[keep]

            trimmed.append(q)

        all_predictions = trimmed

        fig = plot_eos_trend(all_predictions)

        fig_path = OUT_DIR / f"eval6_{SSP}_eos_through_2100.png"

        fig.savefig(fig_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

        print("Saved", fig_path)

        print(f"Done in {time.time() - t_all:.1f}s (cached) | models={MODEL_ORDER}")

        return


    z = np.load(CACHE_EXTRACT)

    

    if CACHE_L.name.endswith(".csv"):

        L_df = pd.read_csv(CACHE_L)

        if "L_star" in L_df.columns:

            L_stars = L_df["L_star"].to_numpy(float)

        elif "preseason_length" in L_df.columns:

            L_stars = L_df["preseason_length"].to_numpy(float)

        elif "preseason_p_months" in L_df.columns:

                                            

            L_stars = L_df["preseason_p_months"].to_numpy(float) * 30.0

        else:

            L_stars = np.full(len(z["latitude"]), 60.0)

    else:

        L_stars = np.full(len(z["latitude"]), 60.0)


    if RIDGE_FP.exists():

        ridge = pd.read_csv(RIDGE_FP)

    else:

        ridge = pd.DataFrame()


    n = len(z["latitude"])

    daily_t = np.asarray(z["daily_t"])

    daily_p = np.asarray(z["daily_p"]) if "daily_p" in z.files else None

    meta = []

    for i in range(n):

        lat, lon = float(z["latitude"][i]), float(z["longitude"][i])

        reg = fig3_region_from_ridge(lat, lon, ridge)

        m_eos = float(np.nanmean(z["eos"][i])) if "eos" in z else 270.0

        l_val = float(L_stars[i]) if np.isfinite(L_stars[i]) else 60.0

        dt = as_doy_year(daily_t[i])

        n_clim = int(np.isfinite(dt).any(axis=0).sum())

        meta.append({

            "pixel": i, "latitude": lat, "longitude": lon, "region": reg,

            "L_star": l_val,

            "mean_eos": m_eos,

            "n_valid": int(np.isfinite(z["eos"][i]).sum()) if "eos" in z else 30,

            "n_clim": n_clim,

        })

    pix = pd.DataFrame(meta)


    elig = pix[(pix.n_valid >= 10) & (pix.n_clim >= 10)].copy()

    pix_s = elig.reset_index(drop=True)

    print(

        f"Eligible grids: {len(pix_s)} / {len(pix)} "

        f"(by region: {pix_s.groupby('region').size().to_dict()})"

    )


    model_list = get_model_list()

    years_hist = np.asarray(z["years"] if "years" in z else range(1982, 2023), int)

    eos_all = np.asarray(z["eos"])

    sos_all = np.asarray(z["sos"])

    preseason_p = np.asarray(z["preseason_p"]) if "preseason_p" in z.files else None

    preseason_t = np.asarray(z["preseason_t"]) if "preseason_t" in z.files else None


    print(f"Climate extract: {CACHE_EXTRACT.name} (n={n})")

    print(

        f"Fitting {len(pix_s)} pixels × {MODEL_ORDER} | MAXITER={MAXITER} | "

        f"N_JOBS={N_JOBS} | full-sample (no CV)"

    )


    job_args = []

    for _, row in pix_s.iterrows():

        pi = int(row.pixel)

        job_args.append(dict(

            r=row.to_dict(),

            daily_t_pi=daily_t[pi],

            daily_p_pi=None if daily_p is None else daily_p[pi],

            eos=eos_all[pi],

            sos=sos_all[pi],

            preseason_p_pi=None if preseason_p is None else preseason_p[pi],

            preseason_t_pi=None if preseason_t is None else preseason_t[pi],

            years_hist=years_hist,

            model_list=model_list,

        ))


    fits_raw = Parallel(n_jobs=N_JOBS, prefer="processes")(

        delayed(fit_one_pixel)(**kw) for kw in job_args

    )

    fits = [f for f in fits_raw if f is not None]

    print(f"Fits done: {len(fits)}/{len(pix_s)} in {time.time() - t_all:.1f}s")


    print(f"Loading {SSP} future variables...")

    Tfut = load_future_var(TABLES / SSP / "daily_min_t_clipped", f"daily_min_t_clipped_2025-2100_{SSP}_pixel_*.csv", "tasmin")

    Pfut = load_future_var(TABLES / SSP / "daily_p", f"daily_p_2025-2100_{SSP}_pixel_*.csv", "pr")


    fut_coords = np.array(list(Tfut.keys()))

    tree = cKDTree(fut_coords)


    all_predictions = []

    diag_rows = []


    n_skip_fut = 0

    for i_f, f in enumerate(fits):

        if (i_f + 1) % 200 == 0 or i_f == 0:

            print(f"  projecting {i_f + 1}/{len(fits)} ...", flush=True)

        site_lat, site_lon = round(f["latitude"], 4), round(f["longitude"], 4)

        dist, idx = tree.query([site_lat, site_lon])

        if float(dist) > 0.01:

            n_skip_fut += 1

            continue

        key = tuple(fut_coords[idx])


        pi = f["pixel"]

        T_hist = as_doy_year(daily_t[pi])

        P_hist = as_doy_year(daily_p[pi]) if daily_p is not None else np.zeros_like(T_hist)

        T_bc = bias_correct_T(Tfut[key], T_hist)

        P_bc = bias_correct_P(Pfut[key], P_hist)

        sos_fut = project_sos_clipped(f["years_hist"], f["sos_hist"], YEARS_FUT)

        sos_mu, sos_sd = np.nanmean(f["sos_hist"]), np.nanstd(f["sos_hist"])

        if not np.isfinite(sos_sd) or sos_sd < 1e-9: sos_sd = 1.0

        sos_z_fut = (sos_fut - sos_mu) / sos_sd


        P_pre = preseason_p_sum_one(P_bc, f["mean_eos"], f["L_star"])

        P_z = (P_pre - f["P_mu"]) / f["P_sd"]

        

        T_pre = np.nanmean(T_bc[150:270, :], axis=0)

        T_z = (T_pre - f["T_mu"]) / f["T_sd"]


        year_idx = np.arange(T_bc.shape[1])

        preds = {}

        for mname, minfo in model_list.items():

            data = pack_data(T_bc, f["latitude"], year_idx, P_z=P_z, T_z=T_z)

            preds[mname] = np.asarray(

                predict(minfo, f["params"][mname], sos_z_fut, data), float

            )

            all_predictions.append({

                "latitude": f["latitude"], "longitude": f["longitude"], "region": f["region"],

                "model": mname, "predicted_eos": preds[mname], "years": YEARS_FUT,

            })


        diag_rows.append({

            "region": f["region"], "lat": f["latitude"], "lon": f["longitude"],

            **{f"early_{m}": float(np.nanmean(preds[m][EARLY])) for m in MODEL_ORDER},

            **{f"dEOS_{m}": float(np.nanmean(preds[m][LATE]) - np.nanmean(preds[m][EARLY])) for m in MODEL_ORDER},

        })


    print(f"Projected {len(diag_rows)} grids; skipped (no future match): {n_skip_fut}")

    diag = pd.DataFrame(diag_rows)

    diag_path = OUT_DIR / f"eval6_{SSP}_diagnostics.csv"

    diag.to_csv(diag_path, index=False)

    print("Saved", diag_path)


    fits_path = OUT_DIR / f"eval6_{SSP}_fits.pkl"

    pd.to_pickle(fits, fits_path)

    print("Saved", fits_path)


    pred_path = OUT_DIR / f"eval6_{SSP}_predictions.pkl"

    pd.to_pickle(all_predictions, pred_path)

    print("Saved", pred_path, f"({len(all_predictions)} records)")


    fig = plot_eos_trend(all_predictions)

    fig_path = OUT_DIR / f"eval6_{SSP}_eos_through_2100.png"

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print("Saved", fig_path)

    print(f"Done in {time.time() - t_all:.1f}s | models={MODEL_ORDER}")


if __name__ == "__main__":

    main()

