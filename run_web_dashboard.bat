@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
  "venv\Scripts\python.exe" -m web
) else (
  echo 未找到 venv，请先创建虚拟环境并安装依赖: pip install fastapi uvicorn
  pause
  exit /b 1
)
