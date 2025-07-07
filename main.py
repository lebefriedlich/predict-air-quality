from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Setup logging
if not os.path.exists("logs"):
    os.makedirs("logs")

logger = logging.getLogger("flask_logger")
logger.setLevel(logging.INFO)
handler = TimedRotatingFileHandler(
    filename="logs/flask_app.log",
    when="midnight",
    interval=1,
    backupCount=7,
    encoding="utf-8"
)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = Flask(__name__)

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def categorize(pm25):
    pm25 = safe_float(pm25)
    if pm25 is None:
        return "Tidak Diketahui"
    if pm25 <= 50:
        return "Baik"
    elif pm25 <= 100:
        return "Sedang"
    elif pm25 <= 150:
        return "Tidak Sehat"
    elif pm25 <= 200:
        return "Sangat Tidak Sehat"
    else:
        return "Berbahaya"

def analyze_features(X_raw, y_raw):
    df = pd.DataFrame(X_raw, columns=['pm25', 't', 'h', 'p', 'w', 'dew'])
    df['target'] = y_raw
    logger.info("Korelasi fitur terhadap target:\n%s", df.corr()['target'].drop('target').to_string())

    X_df = df.drop(columns='target')
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_df.columns
    vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
    logger.info("VIF tiap fitur:\n%s", vif_data.to_string(index=False))

def remove_high_vif_features(X_df, threshold=10.0):
    while True:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_df.columns
        vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

        max_vif = vif_data["VIF"].max()
        if max_vif > threshold:
            feature_to_drop = vif_data.loc[vif_data["VIF"].idxmax(), "feature"]
            logger.info("Menghapus fitur dengan VIF tinggi: %s (VIF=%.2f)", feature_to_drop, max_vif)
            X_df = X_df.drop(columns=[feature_to_drop])
        else:
            break
    logger.info("Fitur setelah penghapusan VIF tinggi: %s", list(X_df.columns))
    return X_df

def tune_svr(X, y):
    try:
        param_grid = {
            'C': [1000],
            'epsilon': [0.1, 0.2],
            'gamma': [0.001, 0.01]
        }
        grid = GridSearchCV(SVR(kernel='rbf'), param_grid, scoring='r2', cv=3, n_jobs=1)
        grid.fit(X, y)
        logger.info("Best SVR params: %s", grid.best_params_)
        return grid.best_estimator_
    except Exception as e:
        logger.exception("GridSearchCV gagal: %s", str(e))
        return SVR()

def evaluate_model(X, y):
    try:
        logger.info("Memulai evaluasi model...")
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        model = tune_svr(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        logger.info("RMSE regresi: %.4f", rmse)
        logger.info("R² regresi: %.4f", r2)

        return model
    except Exception as e:
        logger.exception("Gagal evaluasi model: %s", str(e))
        return None

def predict_region(region: dict):
    logger.info("Memproses region: %s", region.get('name'))

    iaqi_data = [
        {k: safe_float(d.get(k)) for k in ['pm25', 't', 'h', 'p', 'w', 'dew']}
        for d in region.get('iaqi', []) if safe_float(d.get('pm25')) is not None
    ]

    logger.info("Total data valid untuk region %s: %d", region.get('name'), len(iaqi_data))

    if len(iaqi_data) < 5:
        logger.warning("Region %s memiliki data kurang dari 5, dilewati.", region.get('name'))
        return {"region_id": region.get('id'), "error": "Data tidak cukup"}

    try:
        df = pd.DataFrame(iaqi_data)
        X_df_original = df[['t', 'h', 'p', 'w', 'dew']]
        imp_initial = SimpleImputer(strategy='mean')
        X_df_imputed = pd.DataFrame(imp_initial.fit_transform(X_df_original), columns=X_df_original.columns)

        X_df_reduced = remove_high_vif_features(X_df_imputed)

        imp_final = SimpleImputer(strategy='mean')  # Refit setelah reduksi fitur
        X_df_reduced = pd.DataFrame(imp_final.fit_transform(X_df_reduced), columns=X_df_reduced.columns)

        X_df_reduced = pd.DataFrame(imp.fit_transform(X_df_reduced), columns=X_df_reduced.columns)

        y = df['pm25'].shift(-1).dropna().values
        analyze_features(np.hstack([df[['pm25']].values[:-1], X_df_reduced.values[:-1]]), y)
        X_raw = X_df_reduced.values[:-1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        pca = PCA(n_components=min(0.95, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        logger.info("Total explained variance by PCA: %.2f%%", pca.explained_variance_ratio_.sum() * 100)

        reg = evaluate_model(X_pca, y)
        if reg is None:
            raise ValueError("Gagal evaluasi model")

        base_date = datetime.strptime(region['date_now'], "%Y-%m-%d")
        df_last = df.tail(24)
        last = {k: df_last[k].mean() for k in X_df_reduced.columns}

        base_input_df = pd.DataFrame([last])
        base_input = imp.transform(base_input_df)

        delta_t = df['t'].diff().mean() if df['t'].count() > 1 else 0.5
        delta_w = df['w'].diff().mean() if df['w'].count() > 1 else 0.2
        delta_dew = df['dew'].diff().mean() if df['dew'].count() > 1 else 0.3

        predictions = []
        for day_offset in range(1, 4):
            pred_date = base_date + timedelta(days=day_offset)
            modified_raw = base_input.copy()
            for i, col in enumerate(X_df_reduced.columns):
                if col == 't':
                    modified_raw[0][i] += delta_t * day_offset
                elif col == 'w':
                    modified_raw[0][i] += delta_w * day_offset
                elif col == 'dew':
                    modified_raw[0][i] += delta_dew * day_offset

            X_scaled_day = scaler.transform(modified_raw)
            X_pca_day = pca.transform(X_scaled_day)
            pred_aqi = reg.predict(X_pca_day)[0]
            pred_class = categorize(pred_aqi)

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_aqi": round(pred_aqi, 2),
                "predicted_category": pred_class
            })

        logger.info("Prediksi selesai untuk region %s", region['name'])
        return {"region_id": region.get('id'), "predictions": predictions}

    except Exception as e:
        logger.exception("Terjadi kesalahan saat memproses region %s: %s", region.get('name'), str(e))
        return {"region_id": region.get('id'), "error": "Terjadi kesalahan saat prediksi"}

@app.route("/")
def index():
    return jsonify({"status": "API is running"})

@app.route("/predict-single-region", methods=["POST"])
def predict_single_region():
    data = request.get_json()
    if not data or not isinstance(data, dict):
        logger.warning("Data request kosong atau tidak sesuai format.")
        return jsonify({"error": "Data tidak valid"}), 400

    result = predict_region(data)
    return jsonify(result)