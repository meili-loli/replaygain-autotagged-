FROM python:3.12-slim

# 特征提取依赖 ffmpeg（soundfile/librosa 为回退方案）
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY web/requirements.txt /tmp/req-web.txt
COPY requirements.txt /tmp/req-at.txt
RUN pip install --no-cache-dir -r /tmp/req-web.txt -r /tmp/req-at.txt

COPY . .

ENV RG_HOST=0.0.0.0
ENV RG_HUB_URL=https://raw.githubusercontent.com/meili-loli/replaygain-autotagged-/main/hub/models.json
EXPOSE 8792

CMD ["python", "web/server/app.py"]
