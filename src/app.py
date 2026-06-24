import numpy as np
import pandas as pd
import joblib
import math
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Cấu hình thư mục web
app = Flask(__name__, static_folder='../web', static_url_path='/')
CORS(app)

MODEL_PATH = 'models/coffee_ridge_pipeline.pkl'

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Đã load mô hình Ridge Regression thành công!")
except FileNotFoundError:
    print(f"❌ Không tìm thấy {MODEL_PATH}.")
    model = None

FEATURE_COLUMNS = [
    'thang_sin', 'thang_cos', 'quy_sin', 'quy_cos',
    'Lag_1_Price', 'Lag_2_Price', 'Lag_7_Price', 'Lag_14_Price', 'Lag_30_Price',
    'MA_7_Price', 'MA_30_Price',
    'Volatility_7D', 'Volatility_30D',
    'Rainfall_30D_Sum',
    'ron95_vung1', 'Fuel_Price_Change_7D',
    'temperature_2m_mean'
]

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/predict_simulation', methods=['POST'])
def predict_simulation():
    if model is None:
        return jsonify({"error": "Mô hình chưa được load"}), 500
    try:
        data = request.json
        lag_1 = float(data.get('lag_1_price', 105200))
        fuel  = float(data.get('fuel_price', 23500))
        rain  = float(data.get('rainfall', 120))
        temp  = float(data.get('temperature', 28))
        month = int(data.get('month', 6))

        thang_sin = math.sin(2 * math.pi * month / 12)
        thang_cos = math.cos(2 * math.pi * month / 12)
        quarter = (month - 1) // 3 + 1
        quy_sin = math.sin(2 * math.pi * quarter / 4)
        quy_cos = math.cos(2 * math.pi * quarter / 4)

        input_dict = {
            'thang_sin': thang_sin, 'thang_cos': thang_cos,
            'quy_sin': quy_sin, 'quy_cos': quy_cos,
            'Lag_1_Price': lag_1, 'Lag_2_Price': lag_1 - 500, 'Lag_7_Price': lag_1 - 1500, 
            'Lag_14_Price': lag_1 - 2500, 'Lag_30_Price': lag_1 - 4000,
            'MA_7_Price': lag_1 - 800, 'MA_30_Price': lag_1 - 2000,
            'Volatility_7D': 1200.0, 'Volatility_30D': 1800.0,
            'Rainfall_30D_Sum': rain,
            'ron95_vung1': fuel,
            'Fuel_Price_Change_7D': 0.02,
            'temperature_2m_mean': temp
        }

        df_input = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
        predicted_diff = model.predict(df_input)[0]
        final_price = lag_1 + predicted_diff

        return jsonify({
            "status": "success", "base_price": lag_1, 
            "predicted_price": final_price, "diff": predicted_diff
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

import os

@app.route('/api/current_data', methods=['GET'])
def current_data():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        csv_path = os.path.join(
            BASE_DIR,
            '..',
            'data',
            '3_processed',
            'processed_dak_lak_test.csv'
        )

        df = pd.read_csv(csv_path)

        recent_df = df.tail(7)
        latest_day = recent_df.iloc[-1]

        return jsonify({
            "status": "success",
            "current_price": float(latest_day['gia']),
            "fuel_price": float(latest_day['ron95_vung1']),
            "rainfall": float(latest_day['Rainfall_30D_Sum']),
            "temperature": float(latest_day['temperature_2m_mean']),
            "month": int(latest_day['thang']),
            "chart_labels": recent_df['ngay'].astype(str).tolist(),
            "chart_actual": recent_df['gia'].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
