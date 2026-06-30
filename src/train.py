import sys
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson

# ── CẤU HÌNH ────────────────────────────────────────────────────────────────
TRAIN_FILE = 'data/3_processed/processed_dak_lak_train.csv'
VAL_FILE   = 'data/3_processed/processed_dak_lak_val.csv'
TEST_FILE  = 'data/3_processed/processed_dak_lak_test.csv'

# Danh sách số ngày muốn dự báo (Multi-Horizon)
HORIZONS = [1, 3, 7, 30] 

CANDIDATE_FEATURES = [
    'thang_sin', 'thang_cos', 'quy_sin', 'quy_cos',
    'Lag_1_Price', 'Lag_2_Price', 'Lag_7_Price', 'Lag_14_Price', 'Lag_30_Price',
    'MA_7_Price', 'MA_30_Price',
    'Volatility_7D', 'Volatility_30D',
    'Rainfall_30D_Sum',
    'ron95_vung1', 'Fuel_Price_Change_7D',
    'temperature_2m_mean',
]

BASE_PRICE_COL = 'Lag_1_Price' 
REAL_PRICE_COL = 'gia'

def compute_metrics(y_true_abs: np.ndarray, y_pred_abs: np.ndarray, y_prev_abs: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true_abs, y_pred_abs)
    rmse = np.sqrt(mean_squared_error(y_true_abs, y_pred_abs))
    r2   = r2_score(y_true_abs, y_pred_abs)
    
    diff_true = y_true_abs - y_prev_abs
    diff_pred = y_pred_abs - y_prev_abs
    da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100 if len(diff_true) > 0 else 0.0

    residuals = y_true_abs - y_pred_abs
    dw = durbin_watson(residuals)

    return dict(mae=mae, rmse=rmse, r2=r2, da=da, dw=dw)

def main():
    print("⏳ Đang nạp dữ liệu và huấn luyện mô hình đa khung thời gian...")
    try:
        df_train = pd.read_csv(TRAIN_FILE, parse_dates=['ngay'], index_col='ngay')
        df_val   = pd.read_csv(VAL_FILE,   parse_dates=['ngay'], index_col='ngay')
        df_test  = pd.read_csv(TEST_FILE,  parse_dates=['ngay'], index_col='ngay')
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {e}")
        sys.exit(1)

    # Gộp Train và Val để model học được nhiều dữ liệu nhất có thể cho Production
    df_trainval = pd.concat([df_train, df_val]).sort_index()
    actual_features = [c for c in CANDIDATE_FEATURES if c in df_trainval.columns]
    
    os.makedirs('models', exist_ok=True)
    results_summary = []

    # ── VÒNG LẶP HUẤN LUYỆN 1D, 3D, 7D, 30D ──
    for h in HORIZONS:
        print(f"\n⚙️ Đang xử lý mô hình dự báo: {h} Ngày...")
        target_col = f'Target_{h}D'
        
        # Tạo Target động: Giá tương lai (shift lùi lại) trừ Giá cơ sở (Lag_1)
        df_trainval[target_col] = df_trainval[REAL_PRICE_COL].shift(-(h-1)) - df_trainval[BASE_PRICE_COL]
        df_test[target_col] = df_test[REAL_PRICE_COL].shift(-(h-1)) - df_test[BASE_PRICE_COL]
        
        # Bỏ các dòng NaN ở đuôi tập dữ liệu do hàm shift() sinh ra
        valid_train = df_trainval.dropna(subset=actual_features + [target_col])
        valid_test  = df_test.dropna(subset=actual_features + [target_col])
        
        X_train = valid_train[actual_features].values
        y_train = valid_train[target_col].values
        
        X_test = valid_test[actual_features].values
        
        # --- ĐÃ SỬA LỖI NaN TẠI ĐÂY ---
        y_test_diff = valid_test[target_col].values 
        y_test_prev_abs = valid_test[BASE_PRICE_COL].values
        
        # Thay vì dùng shift() lần nữa gây ra NaN, ta tính ngược lại: Thực tế = Chênh lệch + Giá quá khứ
        y_test_true_abs = y_test_diff + y_test_prev_abs
        # ------------------------------
        
        # 1. Huấn luyện Model
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0))
        ])
        model.fit(X_train, y_train)
        
        # 2. Lưu Model ra file
        model_path = f'models/model_{h}d.pkl'
        joblib.dump(model, model_path)
        print(f"   ✅ Đã lưu {model_path} (Train size: {len(X_train)})")
        
        # 3. Đánh giá trên tập Test
        pred_diff = model.predict(X_test)
        pred_abs = y_test_prev_abs + pred_diff
        
        metrics = compute_metrics(y_test_true_abs, pred_abs, y_test_prev_abs)
        naive_metrics = compute_metrics(y_test_true_abs, y_test_prev_abs, y_test_prev_abs) # Baseline: Dự báo bằng đúng giá hôm nay
        
        results_summary.append({
            'Horizon': f'{h} Ngày',
            'Model MAE': f"{metrics['mae']:,.0f}",
            'Naive MAE': f"{naive_metrics['mae']:,.0f}",
            'Model DA (%)': f"{metrics['da']:.2f}",
            'Naive DA (%)': f"{naive_metrics['da']:.2f}"
        })

    # In kết quả tổng quan
    print("\n📊 BẢNG TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (TEST SET):")
    df_results = pd.DataFrame(results_summary).set_index('Horizon')
    print(df_results.to_string())
    print("\n🎉 HOÀN TẤT! Hệ thống đã sẵn sàng cho API đa khung thời gian.")

if __name__ == "__main__":
    main()