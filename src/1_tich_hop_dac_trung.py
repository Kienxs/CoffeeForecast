import sys
import warnings
import pandas as pd
import numpy as np

# ── CẤU HÌNH ĐƯỜNG DẪN ──────────────────────────────────────────────────────
TRAIN_FILE = 'data/2_interim/replication_dak_lak_train.csv'
VAL_FILE   = 'data/2_interim/replication_dak_lak_val.csv'
TEST_FILE  = 'data/2_interim/replication_dak_lak_test.csv'
FUEL_FILE  = 'data/1_raw/fuel_ron95_daily.csv'

TARGET_COL = 'gia'
OUTPUT_DIR = 'data/3_processed/'

def cyclic_encode(series: pd.Series, period: int, name: str) -> pd.DataFrame:
    return pd.DataFrame({
        f'{name}_sin': np.sin(2 * np.pi * series / period),
        f'{name}_cos': np.cos(2 * np.pi * series / period),
    }, index=series.index)

def main():
    print("⏳ Đang tiến hành gộp và xử lý dữ liệu...")
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_val   = pd.read_csv(VAL_FILE)
        df_test  = pd.read_csv(TEST_FILE)
        df_fuel  = pd.read_csv(FUEL_FILE)
    except FileNotFoundError as e:
        print(f"❌ LỖI: Không tìm thấy file! Chi tiết: {e}")
        sys.exit(1)

    df_train['set_type'] = 'train'
    df_val['set_type']   = 'val'
    df_test['set_type']  = 'test'

    df_coffee = pd.concat([df_train, df_val, df_test], axis=0)
    df_coffee['ngay'] = pd.to_datetime(df_coffee['ngay'], format='mixed')
    df_coffee.sort_values('ngay', inplace=True)
    df_coffee.set_index('ngay', inplace=True)

    df_fuel['date'] = pd.to_datetime(df_fuel['date'], format='mixed')
    df_fuel.set_index('date', inplace=True)

    df_merged = df_coffee.join(df_fuel['ron95_vung1'], how='left')
    df_merged['ron95_vung1'] = df_merged['ron95_vung1'].ffill()

    # --- Feature Engineering ---
    df_merged['Target_Diff'] = df_merged[TARGET_COL].diff()

    thang_encoded = cyclic_encode(df_merged.index.to_series().dt.month, 12, 'thang')
    quy_encoded   = cyclic_encode(df_merged.index.to_series().dt.quarter, 4, 'quy')
    df_merged = df_merged.join(thang_encoded).join(quy_encoded)

    df_merged['Lag_1_Price'] = df_merged[TARGET_COL].shift(1)
    df_merged['Lag_2_Price'] = df_merged[TARGET_COL].shift(2)
    df_merged['Lag_7_Price'] = df_merged[TARGET_COL].shift(7)
    df_merged['Lag_14_Price'] = df_merged[TARGET_COL].shift(14)
    df_merged['Lag_30_Price'] = df_merged[TARGET_COL].shift(30)

    df_merged['MA_7_Price']  = df_merged[TARGET_COL].shift(1).rolling(7).mean()
    df_merged['MA_30_Price'] = df_merged[TARGET_COL].shift(1).rolling(30).mean()

    df_merged['Volatility_7D']  = df_merged[TARGET_COL].shift(1).rolling(7).std()
    df_merged['Volatility_30D'] = df_merged[TARGET_COL].shift(1).rolling(30).std()

    if 'precipitation_sum' in df_merged.columns:
        df_merged['Rainfall_30D_Sum'] = df_merged['precipitation_sum'].shift(1).rolling(30).sum()

    df_merged['Fuel_Price_Change_7D'] = df_merged['ron95_vung1'].shift(1).pct_change(periods=7)

    # --- Làm sạch và Lưu ---
    required_cols_for_dropna = [
        'Target_Diff', 'Lag_30_Price', 'MA_30_Price', 'Volatility_30D', 'Fuel_Price_Change_7D', 'ron95_vung1'
    ]
    df_merged.dropna(subset=required_cols_for_dropna, inplace=True)

    train_final = df_merged[df_merged['set_type'] == 'train'].drop(columns=['set_type']).copy()
    val_final   = df_merged[df_merged['set_type'] == 'val'].drop(columns=['set_type']).copy()
    test_final  = df_merged[df_merged['set_type'] == 'test'].drop(columns=['set_type']).copy()

    train_final.to_csv(f'{OUTPUT_DIR}processed_dak_lak_train.csv')
    val_final.to_csv(f'{OUTPUT_DIR}processed_dak_lak_val.csv')
    test_final.to_csv(f'{OUTPUT_DIR}processed_dak_lak_test.csv')

    print(f"✅ HOÀN THÀNH XỬ LÝ DỮ LIỆU! (Train: {len(train_final)}, Val: {len(val_final)}, Test: {len(test_final)})")

if __name__ == "__main__":
    main()