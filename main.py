from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import numpy as np
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

# Imputasi Missing Values
def impute_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    return imputer.fit_transform(X)

# Penanganan Outliers menggunakan Isolation Forest
def remove_outliers(X, y):
    from sklearn.ensemble import IsolationForest
    isolation_forest = IsolationForest(contamination=0.05)
    outliers = isolation_forest.fit_predict(X)
    
    X_filtered = X[outliers == 1]
    y_filtered = y[outliers == 1]

    if X_filtered.shape[0] != y_filtered.shape[0]:
        logger.error(f"Jumlah sampel setelah penghilangan outliers tidak konsisten! ({X_filtered.shape[0]} vs {y_filtered.shape[0]})")
        return None, None
    
    return X_filtered, y_filtered

# Feature Selection menggunakan RFE
def feature_selection(X, y):
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=6)
    selector = selector.fit(X, y)
    return X[:, selector.support_]

# Model tuning with GridSearchCV for SVR
def tune_svr(X, y): 
    try:   
        param_dist = {
            'C': [0.1, 1, 10, 100, 1000],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        svr = SVR()
        grid_search = GridSearchCV(svr, param_dist, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        
        logger.info("Best SVR params: %s", grid_search.best_params_)
        return grid_search.best_estimator_
    except Exception as e:
        logger.exception("GridSearchCV gagal: %s", str(e))
        return SVR()

# Cross-validation for model evaluation
def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    logger.info(f"Cross-validation scores: {scores}")
    logger.info(f"Average CV score: {scores.mean()}")
    return scores.mean()

# Evaluate model with additional MAE, RMSE, and R²
def evaluate_model(X, y):
    logger.info("Memulai evaluasi model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputasi missing values dan remove outliers
    X_train_imputed = impute_missing_values(X_train)
    X_train_no_outliers, y_train_no_outliers = remove_outliers(X_train_imputed, y_train)

    if X_train_no_outliers is None or y_train_no_outliers is None:
        logger.error("Gagal menghapus outliers atau imputasi data.")
        return None
    
    model = tune_svr(X_train_no_outliers, y_train_no_outliers)

    # Cross-validation
    cross_validate_model(model, X_train_no_outliers, y_train_no_outliers)

    r2_train = r2_score(y_train_no_outliers, model.predict(X_train_no_outliers))
    logger.info(f"Train R²: {r2_train:.4f}")
    
    # Evaluasi model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Test R²: {r2:.4f}")

    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R²: {r2:.4f}")

    return model

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
        X = np.array([[d['pm25'], d['t'], d['h'], d['p'], d['w'], d['dew']] for d in iaqi_data])
        y = np.array([d['pm25'] for d in iaqi_data])

        # Memastikan panjang X dan y konsisten
        if X.shape[0] != len(y):
            logger.error("Jumlah sampel pada X dan y tidak konsisten!")
            return {"region_id": region.get('id'), "error": "Jumlah sampel tidak konsisten"}

        # Feature Selection dan PCA
        X_selected = feature_selection(X, y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        pca = PCA(n_components=4)
        X_pca = pca.fit_transform(X_scaled)

        logger.info("Total explained variance by PCA: %.2f%%", pca.explained_variance_ratio_.sum() * 100)

        reg = evaluate_model(X_pca, y)
        if reg is None:
            raise ValueError("Gagal evaluasi model")

        base_date = datetime.strptime(region['date_now'], "%Y-%m-%d")
        last = iaqi_data[-1]  # Ambil data terakhir yang ada (7 hari yang lalu)
        base_input = np.array([[last['pm25'], last['t'], last['h'], last['p'], last['w'], last['dew']]])

        # Pastikan fitur yang digunakan untuk prediksi sama dengan fitur yang digunakan untuk pelatihan
        if base_input.shape[1] != X_selected.shape[1]:
            logger.error(f"Jumlah fitur pada input prediksi ({base_input.shape[1]}) tidak sesuai dengan jumlah fitur pada pelatihan ({X_selected.shape[1]})")
            return {"region_id": region.get('id'), "error": "Fitur tidak konsisten"}

        predictions = []
        pred_date = base_date + timedelta(days=1)  # Hanya 1 hari ke depan

        X_scaled_day = scaler.transform(base_input)
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