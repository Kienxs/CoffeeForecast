import os
import sys
import logging
import argparse
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, List

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. CẤU HÌNH LOGGING 
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


# KHAI BÁO CLASS HUẤN LUYỆN
class CoffeePricePredictorTrainer:
    def __init__(self, train_path: str, val_path: str, test_path: str, model_save_path: str = "linear_model.pkl"):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_save_path = model_save_path
        
        # Cột mục tiêu
        self.target_col = "gia"
        
        # Các cột không dùng để làm features (Target Leakage hoặc không có ý nghĩa toán học)
        self.drop_cols = ["ngay", "gia_chenh_lech"]
        
        self.pipeline = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Đọc dữ liệu từ file CSV, xử lý lỗi và loại bỏ dữ liệu rỗng (NaN)."""
        try:
            df = pd.read_csv(filepath)
            
            initial_shape = df.shape[0]
            df = df.dropna().reset_index(drop=True)
            dropped_rows = initial_shape - df.shape[0]
            
            if dropped_rows > 0:
                logger.warning(f"Đã tự động xóa {dropped_rows} dòng chứa giá trị rỗng (NaN) trong {filepath}.")
                
            logger.info(f"Đã tải dữ liệu từ {filepath} | Kích thước sẵn sàng train: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.error(f"Không tìm thấy file: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Lỗi khi tải file {filepath}: {e}")
            raise

    def prepare_features_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Tách features (X) và target (y). Mọi cột ngoài drop_cols đều tự động thành Feature."""
        X = df.drop(columns=[self.target_col] + self.drop_cols, errors="ignore")
        y = df[self.target_col]
        return X, y

    def build_pipeline(self) -> Pipeline:
        """Xây dựng Scikit-Learn Pipeline gồm chuẩn hóa và mô hình."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),       # Chuẩn hóa (Z-score) giúp mô hình ko bị thiên lệch bởi số lớn (như usd_vnd ~ 25000)
            ('regressor', LinearRegression())   # Mô hình Linear Regression
        ])
        return pipeline

    def evaluate_model(self, model: Pipeline, X: pd.DataFrame, y: pd.Series, dataset_name: str):
        """Đánh giá và in log các metrics."""
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        r2 = r2_score(y, predictions)

        logger.info(f"--- ĐÁNH GIÁ TRÊN TẬP {dataset_name.upper()} ---")
        logger.info(f"MAE  : {mae:.2f}")
        logger.info(f"RMSE : {rmse:.2f}")
        logger.info(f"R2   : {r2:.4f}\n")

    def train_and_evaluate(self):
        """Luồng thực thi chính (Pipeline Execution)."""
        logger.info("Bắt đầu quy trình huấn luyện mô hình...")

        # 1. Đọc dữ liệu
        df_train = self.load_data(self.train_path)
        df_val = self.load_data(self.val_path)
        df_test = self.load_data(self.test_path)

        # 2. Tiền xử lý
        X_train, y_train = self.prepare_features_targets(df_train)
        X_val, y_val = self.prepare_features_targets(df_val)
        X_test, y_test = self.prepare_features_targets(df_test)

        logger.info(f"Số lượng features đầu vào: {X_train.shape[1]}")
        logger.info(f"Danh sách features: {list(X_train.columns)}")

        # 3. Xây dựng và huấn luyện Pipeline
        self.pipeline = self.build_pipeline()
        logger.info("Đang huấn luyện Linear Regression pipeline...")
        self.pipeline.fit(X_train, y_train)
        logger.info("Huấn luyện hoàn tất!")

        # 4. Đánh giá mô hình
        self.evaluate_model(self.pipeline, X_train, y_train, "Train")
        self.evaluate_model(self.pipeline, X_val, y_val, "Validation")
        self.evaluate_model(self.pipeline, X_test, y_test, "Test")

        # 5. Phân tích trọng số (Hiểu mô hình)
        model = self.pipeline.named_steps['regressor']
        feature_names = X_train.columns
        coefs = pd.DataFrame(
            model.coef_, 
            columns=['Hệ số (Coefficient)'], 
            index=feature_names
        ).sort_values(by='Hệ số (Coefficient)', ascending=False)
        
        logger.info("Top 5 đặc trưng ảnh hưởng tích cực (Làm tăng giá):\n%s", coefs.head(5).to_string())
        logger.info("Top 5 đặc trưng ảnh hưởng tiêu cực (Làm giảm giá):\n%s", coefs.tail(5).to_string())
        
        # Kiểm tra nhanh trọng số của USD_VND nếu nó tồn tại
        if 'usd_vnd' in coefs.index:
            usd_weight = coefs.loc['usd_vnd', 'Hệ số (Coefficient)']
            logger.info(f"👉 Trọng số của Tỷ giá USD/VND: {usd_weight:.2f}")

        # 6. Lưu trữ mô hình
        self.save_model()

    def save_model(self):
        """Lưu model pipeline ra file."""
        if self.pipeline is not None:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(self.pipeline, self.model_save_path)
            logger.info(f"Đã lưu mô hình chuẩn hóa và dự đoán tại: {self.model_save_path}")
        else:
            logger.error("Không có mô hình để lưu. Huấn luyện bị lỗi.")


# 3. HÀM MAIN VÀ CLI ARGS
def main():
    parser = argparse.ArgumentParser(description="Script huấn luyện Linear Regression dự đoán giá cà phê")
    
    parser.add_argument("--train", type=str, default="data/coffee_price_train_features.csv", help="Đường dẫn file train")
    parser.add_argument("--val", type=str, default="data/coffee_price_val_features.csv", help="Đường dẫn file validation")
    parser.add_argument("--test", type=str, default="data/coffee_price_test_features.csv", help="Đường dẫn file test")

    parser.add_argument("--model-out", type=str, default="models/coffee_price_lr_model.pkl", help="Tên file model output (.pkl)")
    
    args = parser.parse_args()

    trainer = CoffeePricePredictorTrainer(
        train_path=args.train,
        val_path=args.val,
        test_path=args.test,
        model_save_path=args.model_out
    )
    
    try:
        trainer.train_and_evaluate()
    except Exception as e:
        logger.critical(f"Chương trình bị ngắt do lỗi nghiêm trọng: {e}")

if __name__ == "__main__":
    main()