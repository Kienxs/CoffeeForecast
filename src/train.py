import sys
import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import warnings

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson

# Tắt cảnh báo để log gọn gàng hơn
warnings.filterwarnings('ignore')

# ── CẤU HÌNH ────────────────────────────────────────────────────────────────
TRAIN_FILE = 'data/3_processed/processed_dak_lak_train.csv'
VAL_FILE   = 'data/3_processed/processed_dak_lak_val.csv'
TEST_FILE  = 'data/3_processed/processed_dak_lak_test.csv'

HORIZONS = [1, 3, 7, 30]

# Đã loại bỏ các Lag gần nhau để giảm nhiễu, LightGBM sẽ tự chọn lọc đặc trưng
CANDIDATE_FEATURES = [
    'thang_sin', 'thang_cos', 'quy_sin', 'quy_cos',
    'Lag_1_Price', 'Lag_7_Price', 'Lag_30_Price',
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

def optimize_and_train_model(X_train, y_train):
    """
    Sử dụng TimeSeriesSplit để dò tìm tham số tối ưu mà không bị rò rỉ dữ liệu tương lai.
    """
    lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    
    # Không gian siêu tham số để dò tìm (Tùy chỉnh theo tài nguyên máy)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'num_leaves': [15, 31, 63],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    # Validation trượt theo thời gian (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        estimator=lgb_model,
        param_distributions=param_dist,
        n_iter=15,          # Thử 15 tổ hợp ngẫu nhiên
        cv=tscv,            # Chỉ dùng Validation quá khứ -> tương lai
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_, random_search.best_params_

def main():
    print("⏳ Đang nạp dữ liệu và huấn luyện mô hình đa khung thời gian với LightGBM...")
    try:
        df_train = pd.read_csv(TRAIN_FILE, parse_dates=['ngay'], index_col='ngay')
        df_val   = pd.read_csv(VAL_FILE,   parse_dates=['ngay'], index_col='ngay')
        df_test  = pd.read_csv(TEST_FILE,  parse_dates=['ngay'], index_col='ngay')
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {e}")
        sys.exit(1)

    # Gộp Train và Val để model học. Validation sẽ được thực hiện nội bộ qua TimeSeriesSplit
    df_trainval = pd.concat([df_train, df_val]).sort_index()
    actual_features = [c for c in CANDIDATE_FEATURES if c in df_trainval.columns]
    
    os.makedirs('models', exist_ok=True)
    results_summary = []

    # ── VÒNG LẶP HUẤN LUYỆN 1D, 3D, 7D, 30D ──
    for h in HORIZONS:
        print(f"\n⚙️ Đang xử lý mô hình dự báo: {h} Ngày...")
        target_col = f'Target_{h}D'
        
        # 1. Tạo Target động (Chỉ dùng cho Training/Evaluation)
        df_trainval_h = df_trainval.copy()
        df_test_h = df_test.copy()
        
        df_trainval_h[target_col] = df_trainval_h[REAL_PRICE_COL].shift(-(h-1)) - df_trainval_h[BASE_PRICE_COL]
        df_test_h[target_col] = df_test_h[REAL_PRICE_COL].shift(-(h-1)) - df_test_h[BASE_PRICE_COL]
        
        # 2. Xóa NaN an toàn (Không ảnh hưởng đến tập dữ liệu gốc nếu dùng cho Inference sau này)
        valid_train = df_trainval_h.dropna(subset=actual_features + [target_col])
        valid_test  = df_test_h.dropna(subset=actual_features + [target_col])
        
        X_train = valid_train[actual_features].values
        y_train = valid_train[target_col].values
        
        X_test = valid_test[actual_features].values
        
        # Lấy giá trị cơ sở để phục hồi giá trị thực tế
        y_test_diff = valid_test[target_col].values 
        y_test_prev_abs = valid_test[BASE_PRICE_COL].values
        y_test_true_abs = y_test_diff + y_test_prev_abs
        
        # 3. Huấn luyện Model với TimeSeries C.V
        print("   🔍 Đang tìm kiếm siêu tham số tối ưu...")
        best_model, best_params = optimize_and_train_model(X_train, y_train)
        print(f"   ✨ Tham số tốt nhất: {best_params}")
        
        # 4. Lưu Model ra file
        model_path = f'models/lgbm_model_{h}d.pkl'
        joblib.dump(best_model, model_path)
        print(f"   ✅ Đã lưu {model_path} (Train size: {len(X_train)})")
        
        # 5. Đánh giá trên tập Test
        pred_diff = best_model.predict(X_test)
        pred_abs = y_test_prev_abs + pred_diff
        
        metrics = compute_metrics(y_test_true_abs, pred_abs, y_test_prev_abs)
        naive_metrics = compute_metrics(y_test_true_abs, y_test_prev_abs, y_test_prev_abs) 
        
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
    print("\n🎉 HOÀN TẤT! Mô hình LightGBM đã sẵn sàng cho Production.")

if __name__ == "__main__":
    main()