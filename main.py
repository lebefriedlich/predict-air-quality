from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import numpy as np
from datetime import datetime, timedelta
import logging
from logging.handlers import TimedRotatingFileHandler
import os

# Buat folder logs jika belum ada
if not os.path.exists("logs"):
    os.makedirs("logs")

# Setup logger
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

def categorize(pm25) -> str:
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

def evaluate_model(X, y_class, y_reg):
    try:
        logger.info("Memulai evaluasi model...")

        X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )

        # Model klasifikasi
        clf = SVC(kernel='rbf')
        clf.fit(X_train, y_train_class)
        y_pred_class = clf.predict(X_test)
        acc = accuracy_score(y_test_class, y_pred_class)
        logger.info("Akurasi klasifikasi: %.2f%%", acc * 100)
        logger.info("Laporan klasifikasi:\n%s", classification_report(y_test_class, y_pred_class))

        # Model regresi
        reg = SVR(kernel='rbf')
        reg.fit(X_train, y_train_reg)
        y_pred_reg = reg.predict(X_test)
        rmse = mean_squared_error(y_test_reg, y_pred_reg, squared=False)
        r2 = r2_score(y_test_reg, y_pred_reg)
        logger.info("RMSE regresi: %.4f", rmse)
        logger.info("R² regresi: %.4f", r2)

        return clf, reg

    except Exception as e:
        logger.exception("Gagal evaluasi model: %s", str(e))
        return None, None

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
        return {"Region": region.get('name'), "error": "Data tidak cukup"}

    try:
        X = [[d['pm25'], d['t'], d['h'], d['p'], d['w'], d['dew']] for d in iaqi_data]
        if any(None in row for row in X):
            raise ValueError("Terdapat nilai None dalam data fitur.")

        y_class = [categorize(d['pm25']) for d in iaqi_data]
        y_reg = [d['pm25'] for d in iaqi_data]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        clf, reg = evaluate_model(X_pca, y_class, y_reg)
        if clf is None or reg is None:
            raise ValueError("Gagal evaluasi model")

        base_date = datetime.strptime(region['date_now'], "%Y-%m-%d")
        last = iaqi_data[-1]
        base_input = np.array([[last['pm25'], last['t'], last['h'], last['p'], last['w'], last['dew']]])

        predictions = []
        for day_offset in range(1, 4):
            pred_date = base_date + timedelta(days=day_offset)

            modified_input = base_input.copy()
            modified_input[0][1] += 0.5 * day_offset
            modified_input[0][4] += 0.2 * day_offset
            modified_input[0][5] += 0.3 * day_offset

            X_scaled_day = scaler.transform(modified_input)
            X_pca_day = pca.transform(X_scaled_day)

            pred_aqi = reg.predict(X_pca_day)[0]
            pred_class = clf.predict(X_pca_day)[0]

            predictions.append({
                "date": pred_date.strftime("%Y-%m-%d"),
                "predicted_aqi": round(pred_aqi, 2),
                "predicted_category": pred_class
            })

        logger.info("Prediksi selesai untuk region %s", region['name'])
        return {"region_id": region['id'], "predictions": predictions}

    except Exception as e:
        logger.exception("Terjadi kesalahan saat memproses region %s: %s", region.get('name'), str(e))
        return {"region_id": region.get('id'), "error": "Terjadi kesalahan saat prediksi"}

@app.route("/")
def index():
    return jsonify({"status": "Hello"})

@app.route("/predict-multiple-regions", methods=["POST"])
def predict_multiple_regions():
    data = request.get_json()
    logger.info("Request prediksi diterima, total region: %d", len(data) if data else 0)

    if not data or not isinstance(data, list):
        logger.warning("Data request kosong atau format tidak sesuai.")
        return jsonify({"error": "Data tidak valid"}), 400

    results = [predict_region(region) for region in data]
    logger.info("Prediksi selesai untuk semua region.")
    return jsonify(results)
