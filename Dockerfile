FROM python:3.11-slim

WORKDIR /app

# Copy các file cấu hình
COPY requirements.txt .

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ dự án vào container
COPY . .

# Mở cổng 7860 theo chuẩn Hugging Face
EXPOSE 7860

# Lệnh khởi động app
CMD ["python", "src/app.py"]