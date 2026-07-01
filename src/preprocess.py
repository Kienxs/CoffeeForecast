import pandas as pd

def preprocess_and_encode(train_df, val_df, test_df, fuel_df):
    
    def process_single_df(df_raw):
        df = df_raw.copy()
        df['ngay'] = pd.to_datetime(df['ngay'])
        
        # Gộp và làm sạch
        merged = pd.merge(df, fuel_df, on='ngay', how='left')
        merged = merged.sort_values(by=['thi_truong', 'ngay']).reset_index(drop=True)
        merged['ron95_vung1'] = merged['ron95_vung1'].ffill().bfill()
        merged['is_update_day'] = merged['is_update_day'].fillna(0).astype(int)
        
        cols_to_drop = ['ten_mat_hang', 'coord_source']
        merged.drop(columns=[c for c in cols_to_drop if c in merged.columns], inplace=True)
        
        # Đặc trưng thời gian
        merged['thang'] = merged['ngay'].dt.month
        merged['nam'] = merged['ngay'].dt.year
        return merged

    # Áp dụng xử lý
    train_p = process_single_df(train_df)
    val_p = process_single_df(val_df)
    test_p = process_single_df(test_df)
    
    # Mã hóa One-Hot Encoding
    train_enc = pd.get_dummies(train_p, columns=['thi_truong'], dtype=int)
    val_enc = pd.get_dummies(val_p, columns=['thi_truong'], dtype=int)
    test_enc = pd.get_dummies(test_p, columns=['thi_truong'], dtype=int)
    
    # Đồng bộ cấu trúc cột theo tập Train
    train_columns = train_enc.columns
    val_enc = val_enc.reindex(columns=train_columns, fill_value=0)
    test_enc = test_enc.reindex(columns=train_columns, fill_value=0)
    
    # Đặt Index là ngày tháng
    for df in [train_enc, val_enc, test_enc]:
        df.set_index('ngay', inplace=True)
    

    train_enc.to_csv('coffee_price_train_encoded.csv')
    val_enc.to_csv('coffee_price_val_encoded.csv')
    test_enc.to_csv('coffee_price_test_encoded.csv')

    print("Đã lưu file mã hóa.")
        
    return train_enc, val_enc, test_enc