import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Cấu hình thư mục web
app = Flask(__name__, static_folder='../web', static_url_path='/')
CORS(app)

MODEL_PATH = 'models/coffee_price_lr_model.pkl'
try:
    # Model này là một Pipeline: StandardScaler và LinearRegression
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Đã load thành công mô hình: {MODEL_PATH}")
except FileNotFoundError:
    print(f"⚠️ Cảnh báo: Chưa tìm thấy {MODEL_PATH}. Vui lòng kiểm tra lại đường dẫn.")
    model_pipeline = None

FEATURE_COLUMNS = [
    'temperature_2m_mean', 'temperature_2m_min', 'temperature_2m_max',
    'precipitation_sum', 'rain_sum', 'lat', 'lon', 'ron95_vung1',
    'is_update_day', 'thang', 'nam', 'thi_truong_Gia Lai',
    'thi_truong_Kon Tum', 'thi_truong_Lâm Đồng', 'thi_truong_Đắk Lắk',
    'thi_truong_Đắk Nông', 'lag_1', 'lag_2', 'rolling_mean_3',
    'min_7d', 'max_7d', 'usd_vnd'
]

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Endpoint dự báo mô phỏng
@app.route('/api/predict_simulation', methods=['POST'])
def predict_simulation():
    if model_pipeline is None:
        return jsonify({"error": "Mô hình chưa sẵn sàng"}), 500

    data = request.json
    try:
        # Lấy các tham số chính từ giao diện người dùng
        lag_1 = float(data.get('lag_1_price', 105200))
        fuel  = float(data.get('fuel_price', 23500))
        rain  = float(data.get('rainfall', 10))
        temp  = float(data.get('temperature', 28))
        month = int(data.get('month', 6))
        year  = int(data.get('year', 2026))
        
        # 🔴 Thêm tham số tỷ giá USD (Mặc định lấy 25400 nếu UI chưa gửi lên)
        usd   = float(data.get('usd_vnd', 25400))
        
        # Xử lý chọn tỉnh/thị trường (Mặc định là Đắk Lắk)
        region = data.get('region', 'Đắk Lắk')
        thi_truong = {
            'Gia Lai': 0, 'Kon Tum': 0, 'Lâm Đồng': 0, 'Đắk Lắk': 0, 'Đắk Nông': 0
        }
        if region in thi_truong:
            thi_truong[region] = 1
        else:
            thi_truong['Đắk Lắk'] = 1 # Fallback an toàn

        # 3. TẠO INPUT DICT (Gồm tham số chính + nội suy tham số phụ + usd_vnd)
        input_dict = {
            'temperature_2m_mean': temp,
            'temperature_2m_min': temp - 3.5, 
            'temperature_2m_max': temp + 4.0, 
            'precipitation_sum': rain,
            'rain_sum': rain,
            'lat': 12.66, 
            'lon': 108.03,
            'ron95_vung1': fuel,
            'is_update_day': 1, 
            'thang': month,
            'nam': year,
            'thi_truong_Gia Lai': thi_truong['Gia Lai'],
            'thi_truong_Kon Tum': thi_truong['Kon Tum'],
            'thi_truong_Lâm Đồng': thi_truong['Lâm Đồng'],
            'thi_truong_Đắk Lắk': thi_truong['Đắk Lắk'],
            'thi_truong_Đắk Nông': thi_truong['Đắk Nông'],
            
            # Khối dữ liệu chuỗi thời gian
            'lag_1': lag_1,
            'lag_2': lag_1 - 300,            
            'rolling_mean_3': lag_1 - 100,   
            'min_7d': lag_1 - 1500,          
            'max_7d': lag_1 + 500,
            
            'usd_vnd': usd
        }

        # Chuyển đổi thành DataFrame với đúng thứ tự cột
        df_input = pd.DataFrame([input_dict], columns=FEATURE_COLUMNS)
        
        # Dự đoán GIÁ TRỊ THỰC
        predicted_price = model_pipeline.predict(df_input)[0]
        predicted_diff = predicted_price - lag_1

        return jsonify({
            "status": "success", 
            "base_price": lag_1, 
            "predicted_price": round(predicted_price, 2), 
            "diff": round(predicted_diff, 2),
            "region": region
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/api/current_data', methods=['GET'])
def current_data():
    try:
        # Load file test mới nhất để hiển thị biểu đồ
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, '..', 'data', 'coffee_price_test_features.csv')
        df = pd.read_csv(csv_path)

        # Lọc lấy riêng dữ liệu của 1 tỉnh để vẽ biểu đồ
        df_daklak = df[df['thi_truong_Đắk Lắk'] == 1]
        
        recent_df = df_daklak.tail(7)
        latest_day = recent_df.iloc[-1]

        return jsonify({
            "status": "success",
            "current_price": float(latest_day['gia']),
            "fuel_price": float(latest_day['ron95_vung1']),
            "rainfall": float(latest_day['precipitation_sum']),
            "temperature": float(latest_day['temperature_2m_mean']),
            "month": int(latest_day['thang']),
            "year": int(latest_day['nam']),
            # 🔴 Đọc tỷ giá USD_VND thực tế từ file CSV
            "usd_vnd": float(latest_day.get('usd_vnd', 25400)), 
            "chart_labels": recent_df['ngay'].astype(str).tolist() if 'ngay' in recent_df else [],
            "chart_actual": recent_df['gia'].tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)