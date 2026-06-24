import sys
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.stattools import durbin_watson

# ── CẤU HÌNH ────────────────────────────────────────────────────────────────
TRAIN_FILE = 'data/3_processed/processed_dak_lak_train.csv'
VAL_FILE   = 'data/3_processed/processed_dak_lak_val.csv'
TEST_FILE  = 'data/3_processed/processed_dak_lak_test.csv'

CANDIDATE_FEATURES = [
    'thang_sin', 'thang_cos', 'quy_sin', 'quy_cos',
    'Lag_1_Price', 'Lag_2_Price', 'Lag_7_Price', 'Lag_14_Price', 'Lag_30_Price',
    'MA_7_Price', 'MA_30_Price',
    'Volatility_7D', 'Volatility_30D',
    'Rainfall_30D_Sum',
    'ron95_vung1', 'Fuel_Price_Change_7D',
    'temperature_2m_mean',
]

TARGET_COL = 'Target_Diff' 
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

    return dict(mae=mae, rmse=rmse, r2=r2, da=da, dw=dw, residuals=residuals)

def walk_forward_cv(df: pd.DataFrame, features: list, n_splits: int = 5) -> dict:
    X = df[features].values
    y_diff = df[TARGET_COL].values
    y_prev = df[BASE_PRICE_COL].values
    y_true_abs = df[REAL_PRICE_COL].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for train_idx, val_idx in tscv.split(X):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0))
        ])
        model.fit(X[train_idx], y_diff[train_idx])
        pred_abs = y_prev[val_idx] + model.predict(X[val_idx])
        fold_metrics.append(compute_metrics(y_true_abs[val_idx], pred_abs, y_prev[val_idx]))

    avg = {k: np.mean([fm[k] for fm in fold_metrics]) for k in ['mae', 'rmse', 'r2', 'da', 'dw']}
    return avg

def main():
    print("⏳ Đang nạp dữ liệu và huấn luyện mô hình...")
    try:
        df_train = pd.read_csv(TRAIN_FILE, parse_dates=['ngay'], index_col='ngay')
        df_val   = pd.read_csv(VAL_FILE,   parse_dates=['ngay'], index_col='ngay')
        df_test  = pd.read_csv(TEST_FILE,  parse_dates=['ngay'], index_col='ngay')
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {e}")
        sys.exit(1)

    df_trainval = pd.concat([df_train, df_val]).sort_index()
    actual_features = [c for c in CANDIDATE_FEATURES if c in df_trainval.columns]
    
    df_trainval.dropna(subset=actual_features + [TARGET_COL, REAL_PRICE_COL], inplace=True)
    df_test.dropna(subset=actual_features + [TARGET_COL, REAL_PRICE_COL], inplace=True)

    y_test_abs = df_test[REAL_PRICE_COL].values
    y_test_prev = df_test[BASE_PRICE_COL].values
    
    # Baseline
    naive_metrics = compute_metrics(y_test_abs, y_test_prev, y_test_prev)
    
    # Cross Validation
    cv_metrics = walk_forward_cv(df_trainval, actual_features, n_splits=5)

    # Train Final Model
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))
    ])
    final_model.fit(df_trainval[actual_features], df_trainval[TARGET_COL])
    
    # LƯU MÔ HÌNH CHO API
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/coffee_ridge_pipeline.pkl')
    print("✅ Đã xuất mô hình ra: models/coffee_ridge_pipeline.pkl")

    test_pred_abs = y_test_prev + final_model.predict(df_test[actual_features])
    test_metrics = compute_metrics(y_test_abs, test_pred_abs, y_test_prev)

    # In kết quả tinh gọn
    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (TEST SET):")
    df_compare = pd.DataFrame([
        {'Model': 'Naive Baseline', 'MAE': f"{naive_metrics['mae']:,.0f}", 'DA (%)': f"{naive_metrics['da']:.2f}"},
        {'Model': 'Ridge Regression', 'MAE': f"{test_metrics['mae']:,.0f}", 'DA (%)': f"{test_metrics['da']:.2f}"},
    ]).set_index('Model')
    print(df_compare.to_string())

    # Vẽ biểu đồ (Chạy ngầm lưu ra file)
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.3)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df_test.index, y_test_abs, label="Thực tế", color='#212121', linewidth=2)
    ax1.plot(df_test.index, test_pred_abs, label="AI Ridge", color='#E53935', linestyle='--', linewidth=2)
    ax1.set_title("Dự Báo vs Thực Tế (Test Set)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = fig.add_subplot(gs[1])
    ridge_coefs = final_model.named_steps['ridge'].coef_
    coefs = pd.Series(ridge_coefs, index=actual_features).sort_values(key=abs)
    ax2.barh(coefs.index, coefs.values, color=['#E53935' if v < 0 else '#1976D2' for v in coefs])
    ax2.set_title("Trọng Số Biến (Feature Importance)")
    ax2.grid(True, axis='x', linestyle='--', alpha=0.4)

    os.makedirs('reports', exist_ok=True) # Thêm dòng này để code tự tạo thư mục reports
    plt.savefig('reports/model_evaluation_ridge.png', dpi=100, bbox_inches='tight')
    print("✅ Đã lưu biểu đồ: reports/model_evaluation_ridge.png\n")

if __name__ == "__main__":
    main()