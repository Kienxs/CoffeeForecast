import pandas as pd
import os

def process_and_merge_usd():
    DATA_DIR = "data"
    
    # 1. TIỀN XỬ LÝ FILE USD_VND
    usd_path = os.path.join(DATA_DIR, 'USD_VND.csv')
    print(f"Đang đọc và xử lý dữ liệu {usd_path}...")
    try:
        df_usd = pd.read_csv(usd_path)
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy file {usd_path}. Vui lòng kiểm tra lại.")
        return

    # Chuẩn hóa cột Ngày 
    df_usd['Ngày'] = pd.to_datetime(df_usd['Ngày'], format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    
    # Xử lý cột "Lần cuối": Xóa dấu phẩy và ép về kiểu số thực (float)
    df_usd['usd_vnd'] = df_usd['Lần cuối'].astype(str).str.replace(',', '').astype(float)
    
    # Chỉ trích xuất 2 cột cần thiết và đổi tên cột Ngày cho khớp với file Cà phê
    df_usd = df_usd[['Ngày', 'usd_vnd']].rename(columns={'Ngày': 'ngay'})
    
    # 2. GHÉP NỐI VÀO CÁC FILE CÀ PHÊ
    files_to_process = [
        'coffee_price_train_features.csv',
        'coffee_price_val_features.csv',
        'coffee_price_test_features.csv'
    ]
    
    for filename in files_to_process:
        file_path = os.path.join(DATA_DIR, filename)
        print(f"\nĐang ghép nối dữ liệu vào {file_path}...")
        try:
            df_coffee = pd.read_csv(file_path)
            
            # Sắp xếp lại theo ngày để đảm bảo việc điền dữ liệu (Fill) chính xác
            df_coffee = df_coffee.sort_values('ngay').reset_index(drop=True)
            
            # Hợp nhất dữ liệu (Left Join)
            df_merged = pd.merge(df_coffee, df_usd, on='ngay', how='left')
            
            # Xử lý những ngày ngoại hối nghỉ giao dịch (Thứ 7, CN, Lễ)
            df_merged['usd_vnd'] = df_merged['usd_vnd'].ffill().bfill()
            
            # Ghi đè lại vào file CSV cũ trong thư mục data/
            df_merged.to_csv(file_path, index=False)
            
            print(f"✅ Đã thêm cột 'usd_vnd' thành công!")
            print(df_merged[['ngay', 'gia', 'usd_vnd']].head(3))
            
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy file {file_path}, bỏ qua.")

if __name__ == "__main__":
    process_and_merge_usd()