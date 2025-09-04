# app.py — Prediksi AQI H+1 dengan SVR (tanpa fallback), baseline dihitung sebagai pembanding

from flask import Flask, request, jsonify
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz, logging, os, warnings
from logging.handlers import TimedRotatingFileHandler

warnings.filterwarnings("ignore")

# =========================
# Konfigurasi
# =========================
HORIZON = 1                    # H+1 saja
MIN_SAMPLES_BASE = 20          # minimal sampel setelah fitur jadi
TIME_COL_CANDIDATES = ["ts","timestamp","time","datetime","date","created_at","observed_at","updated_at"]
JAKARTA = pytz.timezone("Asia/Jakarta")

# =========================
# AQI PM2.5 (US EPA)
# =========================
PM25_BREAKPOINTS = [
    (0.0, 12.0,   0,  50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4,101, 150),
    (55.5,150.4,151, 200),
    (150.5,250.4,201,300),
    (250.5,350.4,301,400),
    (350.5,500.4,401,500),
]
AQI_CATEGORIES = [
    (0, 50,   "Baik"),
    (51,100,  "Sedang"),
    (101,150, "Tidak Sehat bagi Kelompok Sensitif"),
    (151,200, "Tidak Sehat"),
    (201,300, "Sangat Tidak Sehat"),
    (301,500, "Berbahaya"),
]

def aqi_from_pm25(pm25: float):
    if pm25 is None or (isinstance(pm25,float) and np.isnan(pm25)): return None
    pm25 = float(pm25)
    for Clow, Chigh, Ilow, Ihigh in PM25_BREAKPOINTS:
        if Clow <= pm25 <= Chigh:
            return (Ihigh-Ilow)/(Chigh-Clow)*(pm25-Clow)+Ilow
    return 500.0

def categorize_aqi(aqi: float):
    if aqi is None or (isinstance(aqi,float) and np.isnan(aqi)): return "Tidak Diketahui"
    aqi = float(aqi)
    for lo, hi, label in AQI_CATEGORIES:
        if lo <= aqi <= hi: return label
    return "Berbahaya"

def safe_float(x):
    try: return float(x)
    except (TypeError, ValueError): return None

# =========================
# Logging
# =========================
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("flask_logger")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler("logs/flask_app.log", when="midnight", interval=1, backupCount=0, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)

# =========================
# App
# =========================
app = Flask(__name__)

# =========================
# Data utils & agregasi
# =========================
def list_to_dataframe(iaqi_list):
    df = pd.DataFrame(iaqi_list)
    for c in ["pm25","t","h","p","w","dew"]:
        if c not in df.columns: df[c] = np.nan
        df[c] = df[c].apply(safe_float)

    ts, ts_col = None, None
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            tmp = pd.to_datetime(df[c], errors="coerce", utc=True)
            if tmp.notna().sum() > (len(df)*0.5):
                ts, ts_col = tmp, c; break
    if ts is None:
        df["ts"] = pd.date_range(start="2000-01-01", periods=len(df), freq="10T", tz="UTC")
        logger.info("Timestamp tidak ditemukan; pakai range 10 menit.")
    else:
        df["ts"] = ts
        logger.info(f"Timestamp terdeteksi pada kolom '{ts_col}'.")
    df = df.sort_values("ts")
    df = df[~df["pm25"].isna()].copy()
    return df

def aggregate_to_rule_local(df: pd.DataFrame, rule: str):
    dfl = df.copy()
    dfl["ts_local"] = dfl["ts"].dt.tz_convert(JAKARTA)
    dfl = dfl.set_index("ts_local").sort_index()
    agg = dfl.resample(rule).mean(numeric_only=True)
    agg = agg.dropna(subset=["pm25"]).reset_index()
    agg["ts"] = agg["ts_local"].dt.tz_convert("UTC")
    return agg

def try_adaptive_aggregation(df: pd.DataFrame):
    # pilih rule dengan jumlah baris terbanyak
    candidates = []
    for rule in ["1D","6H","1H"]:
        dfx = aggregate_to_rule_local(df, rule)
        candidates.append((rule, dfx, dfx.shape[0]))
    best_rule, best_df, _ = max(candidates, key=lambda x: x[2])
    return best_df, best_rule

def impute_df(df: pd.DataFrame, cols):
    work = df.copy()
    try:
        imputer = KNNImputer(n_neighbors=5, weights="distance")
        work[cols] = imputer.fit_transform(work[cols])
    except Exception as e:
        logger.warning(f"KNNImputer gagal, fallback SimpleImputer: {e}")
        imputer = SimpleImputer(strategy="median")
        work[cols] = imputer.fit_transform(work[cols])
    return work

def add_time_features(df: pd.DataFrame):
    out = df.copy()
    if "ts_local" in out.columns: ts = out["ts_local"]
    else:
        ts = out["ts"].dt.tz_convert(JAKARTA)
        out["ts_local"] = ts
    out["hour"] = ts.dt.hour
    out["dow"]  = ts.dt.dayofweek
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24.0)
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24.0)
    out["dow_sin"]  = np.sin(2*np.pi*out["dow"]/7.0)
    out["dow_cos"]  = np.cos(2*np.pi*out["dow"]/7.0)
    return out

# =========================
# Fitur & adaptasi
# =========================
def choose_windows(n_rows):
    # hemat burn-in saat data pendek; pakai jendela lebih kaya saat data panjang
    if n_rows < 28:   return [1,2], []
    if n_rows < 40:   return [1,2,3], [3]
    if n_rows < 60:   return [1,2,3,5], [3,5]
    if n_rows < 120:  return [1,2,3,6,12], [3,6,12]
    return [1,2,3,6,12,24], [3,6,12,24]

def make_supervised(df: pd.DataFrame, target_col="pm25", lags=None, rolls=None, horizon=1):
    lags  = lags  or [1,2,3,5]
    rolls = rolls or [3,5]
    df = df.copy()

    for L in lags:
        df[f"{target_col}_lag{L}"] = df[target_col].shift(L)
    for W in rolls:
        df[f"{target_col}_rmean{W}"] = df[target_col].rolling(W).mean()
        df[f"{target_col}_rstd{W}"]  = df[target_col].rolling(W).std()
        df[f"{target_col}_rmed{W}"]  = df[target_col].rolling(W).median()
    for col in ["t","h","p","w","dew"]:
        for L in lags:
            df[f"{col}_lag{L}"] = df[col].shift(L)
        for W in rolls:
            df[f"{col}_rmean{W}"] = df[col].rolling(W).mean()

    df[f"{target_col}_tplus{horizon}"] = df[target_col].shift(-horizon)
    df = df.dropna().reset_index(drop=True)
    df = add_time_features(df)

    drop_cols = ["ts","ts_local",target_col,f"{target_col}_tplus{horizon}"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values
    y = df[f"{target_col}_tplus{horizon}"].values
    return X, y, feature_cols

def choose_cv_splits(n_samples):
    return int(max(2, min(5, n_samples//8)))

# =========================
# Transform target & weighting
# =========================
def winsorize_series(y, p=0.01):
    lo, hi = np.quantile(y, [p, 1-p])
    return np.clip(y, lo, hi)

class TargetTransformer:
    """Yeo–Johnson + winsorize untuk stabilisasi target"""
    def __init__(self):
        self.pt = PowerTransformer(method="yeo-johnson", standardize=True)
    def fit(self, y):
        yw = winsorize_series(y, p=0.01)
        self.pt.fit(yw.reshape(-1,1)); return self
    def transform(self, y):
        yw = winsorize_series(y, p=0.01)
        return self.pt.transform(yw.reshape(-1,1)).ravel()
    def inverse(self, y_t):
        yy = self.pt.inverse_transform(np.array(y_t).reshape(-1,1)).ravel()
        return np.maximum(0.0, yy)

def make_recent_weights(n, half_life=90):
    idx = np.arange(n)
    lam = np.log(2)/max(1,half_life)
    w = np.exp(lam*(idx-idx.max()))
    return w

# =========================
# Model & evaluasi
# =========================
def select_features_kbest(X, y, names, k=None):
    if X.shape[1] <= 12: return X, None, names
    k = k or min(40, max(10, X.shape[1]//2))
    selector = SelectKBest(score_func=f_regression, k=k)
    Xs = selector.fit_transform(X, y)
    mask = selector.get_support()
    sel_names = [n for n,m in zip(names,mask) if m]
    return Xs, selector, sel_names

def tune_svr_grid(X, y_t):
    # grid lebih luas biar peluang akurasi naik
    params = {
        "C": [1, 10, 50, 100, 500, 1000],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
        "gamma": ["scale", "auto", 0.01, 0.05, 0.1],
        "kernel": ["rbf"],
    }
    splits = choose_cv_splits(len(y_t))
    tscv = TimeSeriesSplit(n_splits=splits)
    svr = SVR()
    gs = GridSearchCV(svr, params, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=0)
    gs.fit(X, y_t)  # tanpa weight saat grid (stabil)
    logger.info(f"SVR best params (splits={splits}): {gs.best_params_}; best MSE: {-gs.best_score_:.4f}")
    return gs.best_estimator_, splits

def walk_forward_scores_svr(model, X, y_orig, y_t, transformer, use_weights=True):
    splits = choose_cv_splits(len(y_t))
    tscv = TimeSeriesSplit(n_splits=splits)
    r2s, maes, rmses = [], [], []
    sw = make_recent_weights(len(y_t)) if use_weights else np.ones_like(y_t)
    for tr, te in tscv.split(X):
        m = SVR(**model.get_params())
        m.fit(X[tr], y_t[tr], sample_weight=sw[tr])
        pred_t = m.predict(X[te])
        pred = transformer.inverse(pred_t)
        r2s.append(r2_score(y_orig[te], pred))
        maes.append(mean_absolute_error(y_orig[te], pred))
        rmses.append(mean_squared_error(y_orig[te], pred, squared=False))
    return {
        "r2_mean": float(np.mean(r2s)),
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "r2_std": float(np.std(r2s)),
        "mae_std": float(np.std(maes)),
        "rmse_std": float(np.std(rmses)),
        "splits": int(splits),
    }

def walk_forward_scores_baseline(X, y_orig):
    # baseline persistence: prediksi = nilai terakhir di train
    splits = choose_cv_splits(len(y_orig))
    tscv = TimeSeriesSplit(n_splits=splits)
    r2s, maes, rmses = [], [], []
    for tr, te in tscv.split(X):
        pred = np.full_like(y_orig[te], fill_value=y_orig[tr][-1], dtype=float)
        r2s.append(r2_score(y_orig[te], pred))
        maes.append(mean_absolute_error(y_orig[te], pred))
        rmses.append(mean_squared_error(y_orig[te], pred, squared=False))
    return {
        "r2_mean": float(np.mean(r2s)),
        "mae_mean": float(np.mean(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "r2_std": float(np.std(r2s)),
        "mae_std": float(np.std(maes)),
        "rmse_std": float(np.std(rmses)),
        "splits": int(splits),
        "baseline": "persistence"
    }

# =========================
# Training H+1 (SVR-only)
# =========================
def train_model_h1(df):
    cols_num = ["pm25","t","h","p","w","dew"]
    df_imp = impute_df(df, cols_num)
    n_rows = df_imp.shape[0]
    LAGS, ROLLS = choose_windows(n_rows)
    logger.info(f"Adaptive windows -> LAGS={LAGS}, ROLLS={ROLLS}, rows={n_rows}")

    X, y, feat_names = make_supervised(df_imp, "pm25", LAGS, ROLLS, horizon=HORIZON)
    if len(y) < MIN_SAMPLES_BASE:
        raise ValueError(f"Data setelah fitur kurang: {len(y)} < {MIN_SAMPLES_BASE}")

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    Xsel, selector, sel_names = select_features_kbest(Xs, y, feat_names, k=None)

    # transform target
    tt = TargetTransformer().fit(y)
    y_t = tt.transform(y)

    # grid search tanpa weights; fit akhir + CV pakai recent-weight
    model, splits = tune_svr_grid(Xsel, y_t)
    # CV (skala asli)
    wf_svr = walk_forward_scores_svr(model, Xsel, y, y_t, tt, use_weights=True)
    logger.info(f"[H+1] SVR CV  R2={wf_svr['r2_mean']:.3f}  MAE={wf_svr['mae_mean']:.2f}  RMSE={wf_svr['rmse_mean']:.2f}  splits={wf_svr['splits']}")

    # baseline comparison
    wf_base = walk_forward_scores_baseline(Xsel, y)
    logger.info(f"[H+1] BASE CV R2={wf_base['r2_mean']:.3f}  MAE={wf_base['mae_mean']:.2f}  RMSE={wf_base['rmse_mean']:.2f}  splits={wf_base['splits']}")

    # Fit final model di seluruh data dengan sample_weight recent
    sw_full = make_recent_weights(len(y_t))
    model.fit(Xsel, y_t, sample_weight=sw_full)

    return {
        "scaler": scaler,
        "selector": selector,
        "model": model,
        "feature_names": sel_names,
        "lags": LAGS,
        "rolls": ROLLS,
        "tt": tt,
        "cv_metrics_svr": wf_svr,
        "cv_metrics_baseline": wf_base
    }

def build_inference_row(df, bundle):
    X_all, _, _ = make_supervised(df, "pm25", bundle["lags"], bundle["rolls"], horizon=HORIZON)
    if X_all.shape[0] == 0: return None
    x = X_all[-1,:].reshape(1,-1)
    Xs = bundle["scaler"].transform(x)
    if bundle["selector"] is not None:
        Xs = bundle["selector"].transform(Xs)
    return Xs

# =========================
# Pipeline per region (H+1)
# =========================
def predict_region_h1(region: dict):
    name = region.get("name","Unknown")
    rid  = region.get("id")
    logger.info(f"Memproses region: {name}")

    iaqi = region.get("iaqi", [])
    if not isinstance(iaqi,list) or len(iaqi)==0:
        return {"region_id": rid, "error": "Data 'iaqi' kosong atau tidak valid"}

    raw_count = len(iaqi)
    df_raw = list_to_dataframe(iaqi)
    raw_after_clean = df_raw.shape[0]

    # agregasi: pilih resolusi dengan baris terbanyak
    df_agg, used_rule = try_adaptive_aggregation(df_raw)
    agg_count = df_agg.shape[0]
    if agg_count < MIN_SAMPLES_BASE:
        return {
            "region_id": rid,
            "error": f"Data setelah agregasi {used_rule} terlalu pendek",
            "debug": {
                "records_raw": raw_count,
                "rows_after_parse": raw_after_clean,
                "rows_after_agg": agg_count,
                "agg_rule": used_rule,
                "min_required": MIN_SAMPLES_BASE
            }
        }

    df = df_agg
    base_date_local = df["ts_local"].iloc[-1].normalize()

    try:
        bundle = train_model_h1(df)
    except Exception as e:
        return {
            "region_id": rid,
            "error": f"Gagal melatih SVR: {str(e)}",
            "debug": {
                "records_raw": raw_count,
                "rows_after_parse": raw_after_clean,
                "rows_after_agg": agg_count,
                "agg_rule": used_rule
            }
        }

    # inference
    df_imp = impute_df(df.copy(), ["pm25","t","h","p","w","dew"])
    x_inf = build_inference_row(df_imp, bundle)
    if x_inf is None:
        return {"region_id": rid, "error": "Baris inferensi tidak tersedia"}

    pred_t = float(bundle["model"].predict(x_inf)[0])
    pm25_pred = float(bundle["tt"].inverse([pred_t])[0])  # balik ke µg/m³
    pm25_pred = max(0.0, pm25_pred)
    aqi_pred = float(round(aqi_from_pm25(pm25_pred), 2))
    cat = categorize_aqi(aqi_pred)
    pred_date_local = (base_date_local + timedelta(days=HORIZON)).date().isoformat()

    predictions = [{
        "date_local": pred_date_local,
        "predicted_pm25": round(pm25_pred, 2),
        "predicted_aqi": aqi_pred,
        "predicted_category": cat,
        "horizon_day": HORIZON,
        "cv_metrics_svr": bundle["cv_metrics_svr"],        # metrik SVR (skala asli)
        "cv_metrics_baseline": bundle["cv_metrics_baseline"]# pembanding baseline
    }]

    model_info = {
        "model_type": "SVR (rbf) — SVR-only",
        "features_used": int(getattr(bundle["model"], "n_features_in_", 0)),
        "feature_selection": "SelectKBest (adaptive)" if bundle["selector"] else "None",
        "horizons": [HORIZON],
        "data_points_used": int(df.shape[0]),
        "aggregation_rule": used_rule,
        "lags": bundle["lags"],
        "rolls": bundle["rolls"]
    }

    return {
        "region_id": rid,
        "region_name": name,
        "predictions": predictions,
        "model_info": model_info
    }

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return jsonify({"status": "API is running"})

@app.route("/predict-single-region", methods=["POST"])
def predict_single_region():
    try:
        data = request.get_json()
        if not data or not isinstance(data, dict):
            logger.warning("Payload kosong atau bukan JSON object.")
            return jsonify({"error": "Data tidak valid"}), 400
        return jsonify(predict_region_h1(data))
    except Exception as e:
        logger.exception(f"Error /predict-single-region: {e}")
        return jsonify({"error": f"Terjadi kesalahan: {str(e)}"}), 500
