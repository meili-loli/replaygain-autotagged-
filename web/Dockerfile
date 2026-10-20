FROM python:3.12-slim

# 特征提取依赖 ffmpeg（soundfile/librosa 为回退方案）
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 容器内必须绑定 0.0.0.0 端口才能映射出去
ENV RG_HOST=0.0.0.0 RG_PORT=8792
EXPOSE 8792

CMD ["python", "server/app.py"]
