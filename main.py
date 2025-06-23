from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

def categorize(pm25: float) -> str:
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

def predict_region(region: dict):
    iaqi_data = [i for i in region['iaqi'] if i.get('pm25') is not None]
    if len(iaqi_data) < 5:
        return {
            "region_id": region['id'],
            "error": "Data tidak cukup"
        }

    X = [[d['pm25'], d['t'], d['h'], d['p'], d['w'], d['dew']] for d in iaqi_data]
    y_class = [categorize(d['pm25']) for d in iaqi_data]
    y_reg = [d['pm25'] for d in iaqi_data]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    clf = SVC(kernel='rbf')
    clf.fit(X_pca, y_class)

    reg = SVR(kernel='rbf')
    reg.fit(X_pca, y_reg)

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

    return {
        "region_id": region['id'],
        "predictions": predictions
    }

@app.route("/")
def index():
    return jsonify({"status": "Hello"})

@app.route("/predict-multiple-regions", methods=["POST"])
def predict_multiple_regions():
    data = request.get_json()
    results = [predict_region(region) for region in data]
    return jsonify(results)
