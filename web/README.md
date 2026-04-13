# Web 看板

与 **桌面监测程序完全独立**：不安装、不启动本模块时，`python main.py` 照常工作。

## 作用

- 只读 `logs/events.jsonl`、`logs/events.csv`、截图与录像文件
- 浏览器里查看统计、列表、截图与录像下载
- **不运行** YOLO/分类模型，不占用 GPU

## 启动（项目根目录）

```bash
.\venv\Scripts\activate
pip install fastapi uvicorn
python -m web
```

或 Windows：**双击 `run_web_dashboard.bat`**

浏览器：<http://127.0.0.1:8080>  
API 文档：<http://127.0.0.1:8080/docs>

## 与配置一致

日志目录与桌面端相同：读取 `config/settings.json` 中的 `logs.output_dir`（默认 `logs`）。请与桌面使用同一项目目录。

## 安全

默认仅监听 `127.0.0.1`。若要对局域网开放，需自行改 `web/__main__.py` 中的 host，并注意防火墙与鉴权。
