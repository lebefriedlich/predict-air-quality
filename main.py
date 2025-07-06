from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import datetime, timedelta
import logging
from logging.handlers import TimedRotatingFileHandler
import os

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

# Util
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

# Model tuning
def tune_svr(X, y):
    try:
        param_grid = {
            'C': [100],
            'epsilon': [0.1, 0.2],
            'gamma': [0.01]
        }

        grid = GridSearchCV(
            SVR(kernel='rbf'),
            param_grid,
            scoring='r2',
            cv=3,
            n_jobs=1
        )
        grid.fit(X, y)
        logger.info("Best SVR params: %s", grid.best_params_)
        return grid.best_estimator_
    except Exception as e:
        logger.exception("GridSearchCV gagal: %s", str(e))
        return SVR()

def evaluate_model(X, y):
    try:
        logger.info("Memulai evaluasi model...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
        {
            'pm25': safe_float(d.get('pm25')),
            't': safe_float(d.get('t')),
            'h': safe_float(d.get('h')),
            'p': safe_float(d.get('p')),
            'w': safe_float(d.get('w')),
            'dew': safe_float(d.get('dew'))
        }
        for d in region.get('iaqi', [])
        if safe_float(d.get('pm25')) is not None
    ]

    logger.info("Total data valid untuk region %s: %d", region.get('name'), len(iaqi_data))

    if len(iaqi_data) < 5:
        logger.warning("Region %s memiliki data kurang dari 5, dilewati.", region.get('name'))
        return {"region_id": region.get('id'), "error": "Data tidak cukup"}

    try:
        X = [[d['pm25'], d['t'], d['h'], d['p'], d['w'], d['dew']] for d in iaqi_data]
        y = [d['pm25'] for d in iaqi_data]

        if any(None in row for row in X):
            raise ValueError("Terdapat nilai None dalam data fitur.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)

        logger.info("Total explained variance by PCA: %.2f%%", pca.explained_variance_ratio_.sum() * 100)

        reg = evaluate_model(X_pca, y)
        if reg is None:
            raise ValueError("Gagal evaluasi model")

        base_date = datetime.strptime(region['date_now'], "%Y-%m-%d")
        last = iaqi_data[-1]
        base_input = np.array([[last['pm25'], last['t'], last['h'], last['p'], last['w'], last['dew']]])

        predictions = []
        for day_offset in range(1, 4):
            pred_date = base_date + timedelta(days=day_offset)
            modified_input = base_input.copy()
            modified_input[0][1] += 0.5 * day_offset  # suhu
            modified_input[0][4] += 0.2 * day_offset  # angin
            modified_input[0][5] += 0.3 * day_offset  # dew point

            X_scaled_day = scaler.transform(modified_input)
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

# Routes
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
