@echo off
chcp 65001 >nul
title replaygain-nyan
cd /d "%~dp0"

echo ==============================================
echo   replaygain-nyan  本地版
echo   服务运行期间别关这个窗口，关闭 = 停止
echo ==============================================

where python >nul 2>nul
if errorlevel 1 (
  echo [X] 没找到 Python。
  echo     请先安装 Python 3.9+ 并勾选 "Add Python to PATH"：
  echo     https://www.python.org/downloads/
  pause
  exit /b 1
)

python -c "import fastapi, uvicorn, mutagen, multipart, jwt, lightgbm, librosa, soundfile, pyloudnorm, resampy" >nul 2>nul
if errorlevel 1 (
  echo [首次运行] 缺依赖，开始自动安装（可能 5-10 分钟，只需一次）...
  python -m pip install -q -r web\requirements.txt
  if errorlevel 1 ( echo [X] web 依赖安装失败 & pause & exit /b 1 )
  python -m pip install -q -r requirements.txt
  python -m pip install -q resampy
  echo 依赖安装完成 ♡
)

rem 模型市场 hub 索引（本仓库的 hub/models.json）
set RG_HUB_URL=https://raw.githubusercontent.com/meili-loli/replaygain-autotagged-/main/hub/models.json

echo 启动服务... 浏览器稍后自动打开 http://127.0.0.1:8792
start "" cmd /c "timeout /t 4 >nul & start "" http://127.0.0.1:8792"
python web\server\app.py
pause
