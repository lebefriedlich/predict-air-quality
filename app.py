from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import warnings
warnings.filterwarnings('ignore')

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

# Global variables untuk menyimpan preprocessor
global_scaler = None
global_pca = None
global_selectors = None

# Pipeline preprocessing yang komprehensif dengan konsistensi data
def preprocess_pipeline(X, y, fit=True):
    """Pipeline preprocessing yang lengkap dengan feature engineering, scaling, dan PCA"""
    global global_scaler, global_pca, global_selectors
    
    try:
        logger.info(f"Input data shape: X={X.shape}, y={len(y) if y is not None else 'None'}")
        
        # Step 1: Imputasi missing values
        X_imputed = impute_missing_values(X)
        logger.info(f"After imputation: X={X_imputed.shape}")
        
        # Step 2: Remove outliers (hanya pada training)
        if fit and y is not None:
            X_clean, y_clean = remove_outliers(X_imputed, y)
            if X_clean is None or y_clean is None:
                logger.error("Gagal menghapus outliers, menggunakan data asli")
                X_clean, y_clean = X_imputed, y
            logger.info(f"After outlier removal: X={X_clean.shape}, y={len(y_clean)}")
        else:
            X_clean, y_clean = X_imputed, y
        
        # Step 3: Feature engineering
        X_engineered = create_engineered_features(X_clean)
        logger.info(f"After feature engineering: X={X_engineered.shape}")
        
        # Step 4: Feature selection yang disederhanakan
        if fit and y_clean is not None:
            X_selected, selector_kbest, _ = feature_selection(X_engineered, y_clean)
            if X_selected is not None and selector_kbest is not None:
                global_selectors = selector_kbest
                logger.info(f"After feature selection: X={X_selected.shape}")
            else:
                logger.warning("Feature selection gagal atau dilewati, menggunakan semua fitur")
                X_selected = X_engineered
                global_selectors = None
        else:
            if global_selectors is not None:
                try:
                    X_selected = global_selectors.transform(X_engineered)
                    logger.info(f"Applied feature selection: X={X_selected.shape}")
                except Exception as e:
                    logger.warning(f"Gagal menerapkan feature selection: {str(e)}")
                    X_selected = X_engineered
            else:
                X_selected = X_engineered
        
        # Step 5: Scaling dengan RobustScaler (lebih tahan outliers)
        if fit:
            global_scaler = RobustScaler()
            X_scaled = global_scaler.fit_transform(X_selected)
            logger.info(f"After scaling (fit): X={X_scaled.shape}")
        else:
            if global_scaler is not None:
                X_scaled = global_scaler.transform(X_selected)
                logger.info(f"After scaling (transform): X={X_scaled.shape}")
            else:
                X_scaled = X_selected
        
        # Step 6: PCA yang disederhanakan (hanya jika fitur > 10)
        if fit:
            # Hanya gunakan PCA jika fitur terlalu banyak untuk mengurangi overfitting
            if X_scaled.shape[1] > 10:
                try:
                    # Gunakan threshold variance yang lebih konservatif (85%)
                    pca_temp = PCA()
                    pca_temp.fit(X_scaled)
                    cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
                    n_components = np.argmax(cumsum_var >= 0.85) + 1
                    n_components = max(3, min(n_components, min(10, X_scaled.shape[1] - 1)))
                    
                    global_pca = PCA(n_components=n_components)
                    X_pca = global_pca.fit_transform(X_scaled)
                    
                    logger.info(f"PCA: {X_scaled.shape[1]} -> {n_components} komponen")
                    logger.info(f"Total explained variance: {global_pca.explained_variance_ratio_.sum():.3f}")
                    logger.info(f"Final processed data: X={X_pca.shape}, y={len(y_clean) if y_clean is not None else 'None'}")
                    
                    return X_pca, y_clean
                except Exception as e:
                    logger.error(f"PCA gagal: {str(e)}, menggunakan data tanpa PCA")
                    global_pca = None
                    return X_scaled, y_clean
            else:
                # Skip PCA untuk dataset kecil
                logger.info(f"Skip PCA (fitur < 10): X={X_scaled.shape}")
                global_pca = None
                return X_scaled, y_clean
        else:
            if global_pca is not None:
                try:
                    X_pca = global_pca.transform(X_scaled)
                    logger.info(f"Applied PCA: X={X_pca.shape}")
                    return X_pca
                except Exception as e:
                    logger.warning(f"Gagal menerapkan PCA: {str(e)}")
                    return X_scaled
            else:
                return X_scaled
        
    except Exception as e:
        logger.exception(f"Preprocessing pipeline gagal: {str(e)}")
        return None

# Ensemble model dengan SVR dan Random Forest
def create_ensemble_model(X, y):
    """Membuat model SVR tunggal untuk mengurangi overfitting"""
    try:
        logger.info(f"Creating SVR model with data shapes: X={X.shape}, y={len(y)}")
        
        # Validasi konsistensi data
        if X.shape[0] != len(y):
            logger.error(f"Inconsistent data shapes: X={X.shape[0]}, y={len(y)}")
            return None
        
        # Gunakan hanya SVR dengan hyperparameter yang disederhanakan
        svr_model = tune_svr(X, y)
        
        if svr_model is None:
            logger.error("Failed to create SVR model")
            return None
            
        logger.info("SVR model berhasil dibuat")
        return svr_model
        
    except Exception as e:
        logger.exception(f"Gagal membuat SVR model: {str(e)}")
        return None

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

# Imputasi Missing Values dengan KNN Imputer yang lebih robust
def impute_missing_values(X):
    """Imputasi missing values menggunakan KNN Imputer untuk hasil yang lebih akurat"""
    try:
        # Gunakan KNN Imputer untuk imputasi yang lebih sophisticated
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        X_imputed = imputer.fit_transform(X)
        logger.info(f"Imputasi selesai. Shape data: {X_imputed.shape}")
        return X_imputed
    except Exception as e:
        logger.warning(f"KNN Imputer gagal, menggunakan SimpleImputer: {str(e)}")
        # Fallback ke SimpleImputer jika KNN gagal
        imputer = SimpleImputer(strategy='median')
        return imputer.fit_transform(X)

# Penanganan Outliers menggunakan Isolation Forest
def remove_outliers(X, y):
    """Menghapus outliers menggunakan Isolation Forest dengan validasi konsistensi"""
    try:
        from sklearn.ensemble import IsolationForest
        logger.info(f"Removing outliers from data: X={X.shape}, y={len(y) if y is not None else 'None'}")
        
        # Validasi input
        if y is not None and X.shape[0] != len(y):
            logger.error(f"Inconsistent input shapes: X={X.shape[0]}, y={len(y)}")
            return None, None
        
        # Isolation Forest untuk deteksi outliers
        isolation_forest = IsolationForest(
            contamination=0.05,  # 5% data dianggap outliers
            random_state=42,
            n_jobs=-1
        )
        
        # Fit dan prediksi outliers
        outliers = isolation_forest.fit_predict(X)
        
        # Filter data (1 = inlier, -1 = outlier)
        mask = outliers == 1
        X_filtered = X[mask]
        y_filtered = y[mask] if y is not None else None
        
        # Validasi hasil
        if y_filtered is not None and X_filtered.shape[0] != len(y_filtered):
            logger.error(f"Jumlah sampel setelah penghilangan outliers tidak konsisten! ({X_filtered.shape[0]} vs {len(y_filtered)})")
            return None, None
        
        logger.info(f"Outliers removed: {np.sum(~mask)} dari {len(mask)} data")
        logger.info(f"Clean data shapes: X={X_filtered.shape}, y={len(y_filtered) if y_filtered is not None else 'None'}")
        
        return X_filtered, y_filtered
        
    except Exception as e:
        logger.exception(f"Gagal menghapus outliers: {str(e)}")
        return None, None

# Feature Selection yang lebih komprehensif
def feature_selection(X, y, n_features=None):
    """Feature selection sederhana menggunakan SelectKBest saja untuk mengurangi overfitting"""
    try:
        logger.info(f"Feature selection input: X={X.shape}, y={len(y) if y is not None else 'None'}")
        
        # Validasi input
        if y is not None and X.shape[0] != len(y):
            logger.error(f"Inconsistent input shapes in feature_selection: X={X.shape[0]}, y={len(y)}")
            return None, None, None
        
        # Hanya gunakan feature selection jika fitur > 8
        if X.shape[1] <= 8:
            logger.info(f"Skip feature selection (fitur <= 8): X={X.shape}")
            return X, None, None
        
        if n_features is None:
            # Lebih konservatif dalam memilih fitur
            n_features = min(X.shape[1], max(5, X.shape[1] // 3))
        
        # Hanya gunakan SelectKBest untuk kesederhanaan
        selector_kbest = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector_kbest.fit_transform(X, y)
        
        # Validasi hasil
        if X_selected.shape[0] != len(y):
            logger.error(f"Inconsistent output shapes in feature_selection: X={X_selected.shape[0]}, y={len(y)}")
            return None, None, None
        
        logger.info(f"Feature selection completed: {X.shape[1]} -> {X_selected.shape[1]} fitur")
        logger.info(f"Final selected data shape: X={X_selected.shape}")
        
        return X_selected, selector_kbest, None
    except Exception as e:
        logger.error(f"Feature selection gagal: {str(e)}")
        return X, None, None

# Model tuning dengan hyperparameter yang lebih komprehensif
def tune_svr(X, y): 
    """Hyperparameter tuning untuk SVR dengan parameter yang disederhanakan"""
    try:
        logger.info(f"Tuning SVR with data shapes: X={X.shape}, y={len(y)}")
        
        # Validasi konsistensi data
        if X.shape[0] != len(y):
            logger.error(f"Inconsistent data shapes in tune_svr: X={X.shape[0]}, y={len(y)}")
            return None
            
        # Parameter grid yang disederhanakan untuk mengurangi overfitting
        param_dist = {
            'C': [1, 10, 100],  # Kurangi variasi C
            'epsilon': [0.01, 0.1, 0.2],  # Kurangi variasi epsilon
            'gamma': ['scale', 'auto'],  # Hanya gunakan gamma otomatis
            'kernel': ['rbf']  # Hanya gunakan RBF kernel yang paling stabil
        }
        
        # Gunakan TimeSeriesSplit dengan lebih banyak fold untuk validasi yang lebih robust
        tscv = TimeSeriesSplit(n_splits=5)
        
        svr = SVR()
        grid_search = GridSearchCV(
            svr, 
            param_dist, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0  # Kurangi verbosity
        )
        grid_search.fit(X, y)
        
        logger.info("Best SVR params: %s", grid_search.best_params_)
        logger.info("Best SVR score: %.4f", -grid_search.best_score_)
        return grid_search.best_estimator_
    except Exception as e:
        logger.exception("GridSearchCV gagal: %s", str(e))
        # Return SVR dengan parameter default yang conservative
        return SVR(C=10, epsilon=0.1, gamma='scale', kernel='rbf')

# Feature Engineering untuk menciptakan fitur yang lebih informatif
def create_engineered_features(X):
    """Membuat fitur tambahan yang sederhana dan relevan"""
    try:
        # Asumsi X memiliki kolom: [t, h, p, w, dew]
        X_eng = X.copy()
        
        # Hanya tambahkan fitur yang paling relevan untuk mengurangi overfitting
        X_eng = np.column_stack([
            X_eng,
            X[:, 0] - X[:, 4],  # temperature - dew point (comfort index)
            X[:, 0] * X[:, 1]   # temperature * humidity (heat index)
        ])
        
        logger.info(f"Feature engineering: {X.shape[1]} -> {X_eng.shape[1]} fitur")
        return X_eng
    except Exception as e:
        logger.error(f"Feature engineering gagal: {str(e)}")
        return X

# Cross-validation untuk evaluasi model
def cross_validate_model(model, X, y):
    """Cross-validation yang lebih robust dengan lebih banyak fold dan multiple metrics"""
    try:
        # Gunakan lebih banyak fold untuk validasi yang lebih robust
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Evaluasi dengan multiple metrics
        mse_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        mae_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        
        logger.info(f"Cross-validation MSE scores: {mse_scores}")
        logger.info(f"Cross-validation R² scores: {r2_scores}")
        logger.info(f"Cross-validation MAE scores: {mae_scores}")
        
        logger.info(f"Average MSE: {mse_scores.mean():.4f} (±{mse_scores.std():.4f})")
        logger.info(f"Average R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        logger.info(f"Average MAE: {mae_scores.mean():.4f} (±{mae_scores.std():.4f})")
        
        return {
            'mse': mse_scores.mean(),
            'r2': r2_scores.mean(),
            'mae': mae_scores.mean(),
            'mse_std': mse_scores.std(),
            'r2_std': r2_scores.std(),
            'mae_std': mae_scores.std()
        }
    except Exception as e:
        logger.error(f"Cross-validation gagal: {str(e)}")
        return None

def create_baseline_model(X, y):
    """Membuat model Linear Regression sebagai baseline untuk perbandingan"""
    try:
        logger.info(f"Creating baseline Linear Regression model with data shapes: X={X.shape}, y={len(y)}")
        
        # Validasi konsistensi data
        if X.shape[0] != len(y):
            logger.error(f"Inconsistent data shapes in baseline: X={X.shape[0]}, y={len(y)}")
            return None
        
        # Model Linear Regression sederhana
        baseline_model = LinearRegression()
        baseline_model.fit(X, y)
        
        logger.info("Baseline Linear Regression model berhasil dibuat")
        return baseline_model
        
    except Exception as e:
        logger.exception(f"Gagal membuat baseline model: {str(e)}")
        return None

# Evaluasi model dengan ensemble method dan preprocessing yang lebih baik
def evaluate_model(X, y):
    """Evaluasi model dengan baseline comparison dan cross-validation yang diperbaiki"""
    logger.info("Memulai evaluasi model dengan pipeline yang ditingkatkan...")
    
    try:
        # Split data dengan strategi time series
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

        # Pipeline preprocessing
        X_train_processed, y_train_processed = preprocess_pipeline(X_train, y_train, fit=True)
        X_test_processed = preprocess_pipeline(X_test, y_test, fit=False)
        
        if X_train_processed is None or X_test_processed is None:
            logger.error("Preprocessing pipeline gagal")
            return None
        
        # === BASELINE MODEL EVALUATION ===
        logger.info("=== EVALUATING BASELINE MODEL ===")
        baseline_model = create_baseline_model(X_train_processed, y_train_processed)
        if baseline_model is not None:
            baseline_cv = cross_validate_model(baseline_model, X_train_processed, y_train_processed)
            if baseline_cv is not None:
                logger.info(f"Baseline CV R²: {baseline_cv['r2']:.4f} (±{baseline_cv['r2_std']:.4f})")
        
        # === SVR MODEL EVALUATION ===
        logger.info("=== EVALUATING SVR MODEL ===")
        svr_model = create_ensemble_model(X_train_processed, y_train_processed)
        
        if svr_model is None:
            logger.error("Gagal membuat SVR model")
            return None
        
        # Cross-validation dengan multiple metrics
        cv_results = cross_validate_model(svr_model, X_train_processed, y_train_processed)
        
        # Training metrics
        y_train_pred = svr_model.predict(X_train_processed)
        train_r2 = r2_score(y_train_processed, y_train_pred)
        train_mae = mean_absolute_error(y_train_processed, y_train_pred)
        train_rmse = mean_squared_error(y_train_processed, y_train_pred, squared=False)
        
        logger.info(f"SVR Train R²: {train_r2:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}")
        
        # Test metrics
        y_test_pred = svr_model.predict(X_test_processed)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        
        logger.info(f"SVR Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
        
        if cv_results is not None:
            logger.info(f"SVR CV R²: {cv_results['r2']:.4f} (±{cv_results['r2_std']:.4f})")
            
            # Comparison with baseline
            if baseline_cv is not None:
                improvement = cv_results['r2'] - baseline_cv['r2']
                logger.info(f"R² Improvement over baseline: {improvement:.4f}")
                if improvement < 0.05:
                    logger.warning("SVR model tidak memberikan peningkatan signifikan dibanding baseline!")
        
        return svr_model, global_scaler, global_pca, global_selectors
        
    except Exception as e:
        logger.exception(f"Evaluasi model gagal: {str(e)}")
        return None

def predict_region(region: dict):
    """Prediksi kualitas udara untuk region dengan pipeline yang ditingkatkan"""
    logger.info("Memproses region: %s", region.get('name'))

    # Ekstrak data IAQI
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

    # Validasi minimum data
    if len(iaqi_data) < 10:  # Tingkatkan minimum data untuk model yang lebih robust
        logger.warning("Region %s memiliki data kurang dari 10, dilewati.", region.get('name'))
        return {"region_id": region.get('id'), "error": "Data tidak cukup (minimum 10 data points)"}

    try:
        # Persiapan data
        X = np.array([[d['t'], d['h'], d['p'], d['w'], d['dew']] for d in iaqi_data])
        y = np.array([d['pm25'] for d in iaqi_data])

        # Validasi konsistensi data
        if X.shape[0] != len(y):
            logger.error("Jumlah sampel pada X dan y tidak konsisten!")
            return {"region_id": region.get('id'), "error": "Jumlah sampel tidak konsisten"}

        # Validasi data tidak mengandung NaN atau Inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
            logger.warning("Data mengandung NaN atau Inf, akan dibersihkan")
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                return {"region_id": region.get('id'), "error": "Data valid tidak cukup setelah pembersihan"}

        # Training model dengan pipeline yang ditingkatkan
        model_result = evaluate_model(X, y)
        if model_result is None:
            raise ValueError("Gagal evaluasi model")
        
        # Unpack hasil model
        if isinstance(model_result, tuple):
            ensemble_model, scaler, pca, selectors = model_result
        else:
            ensemble_model = model_result
            scaler, pca, selectors = global_scaler, global_pca, global_selectors

        # Persiapan data untuk prediksi
        base_date = datetime.strptime(region['date_now'], "%Y-%m-%d")
        last_data = iaqi_data[-1]  # Data terakhir
        
        # Buat prediksi untuk beberapa hari ke depan (sequential prediction)
        predictions = []
        current_input = last_data.copy()  # Mulai dengan data terakhir
        
        for days_ahead in range(1, 4):  # Prediksi 3 hari ke depan
            pred_date = base_date + timedelta(days=days_ahead)
            
            # Input untuk prediksi (gunakan data dari prediksi sebelumnya atau data terakhir)
            base_input = np.array([[current_input['t'], current_input['h'], current_input['p'], current_input['w'], current_input['dew']]])
            
            # Proses input melalui pipeline preprocessing
            input_processed = preprocess_pipeline(base_input, None, fit=False)
            
            if input_processed is not None:
                # Prediksi dengan ensemble model
                pred_aqi = ensemble_model.predict(input_processed)[0]
                pred_class = categorize(pred_aqi)
                
                predictions.append({
                    "date": pred_date.strftime("%Y-%m-%d"),
                    "predicted_aqi": float(round(max(0, pred_aqi), 2)),  # Konversi ke Python float
                    "predicted_category": pred_class,
                    "confidence": "high" if len(iaqi_data) > 20 else "medium"
                })
                
                # Update input untuk prediksi hari berikutnya
                # Gunakan hasil prediksi sebagai baseline untuk hari berikutnya
                # Asumsi: kondisi meteorologi berubah secara gradual
                current_input = {
                    't': current_input['t'] + np.random.normal(0, 0.5),  # Variasi kecil suhu
                    'h': max(0, min(100, current_input['h'] + np.random.normal(0, 2))),  # Variasi kelembaban (0-100%)
                    'p': current_input['p'] + np.random.normal(0, 1),  # Variasi tekanan
                    'w': max(0, current_input['w'] + np.random.normal(0, 0.3)),  # Variasi kecepatan angin
                    'dew': current_input['dew'] + np.random.normal(0, 0.5)  # Variasi dew point
                }
                
                logger.info(f"Prediksi hari ke-{days_ahead}: AQI={pred_aqi:.2f}, Input selanjutnya: T={current_input['t']:.1f}, H={current_input['h']:.1f}")
            else:
                logger.warning(f"Gagal memproses input untuk prediksi hari ke-{days_ahead}")
                break  # Hentikan prediksi jika ada error

        if not predictions:
            return {"region_id": region.get('id'), "error": "Gagal membuat prediksi"}

        logger.info("Prediksi selesai untuk region %s dengan %d prediksi", region['name'], len(predictions))
        return {
            "region_id": region.get('id'), 
            "predictions": predictions,
            "model_info": {
                "data_points": int(len(iaqi_data)),
                "model_type": "SVR + Random Forest Ensemble",
                "features_used": int(X.shape[1] if global_pca is None else global_pca.n_components_),
                "pca_variance_explained": float(global_pca.explained_variance_ratio_.sum()) if global_pca else None
            }
        }

    except Exception as e:
        logger.exception("Terjadi kesalahan saat memproses region %s: %s", region.get('name'), str(e))
        return {"region_id": region.get('id'), "error": f"Terjadi kesalahan saat prediksi: {str(e)}"}

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