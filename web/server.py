"""
轻量 Web 看板 — 仅只读 logs，不运行检测，与桌面端完全独立进程。

未安装 fastapi 时，桌面程序 main.py 不受影响（本模块不会被导入）。

启动（项目根目录）:
    python -m web
    或: uvicorn web.server:app --host 127.0.0.1 --port 8080
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.paths import PROJECT_ROOT

# 延迟导入 settings，避免在无 config 时崩溃
def _logs_base() -> Path:
    try:
        from utils.settings import get_settings

        rel = (get_settings().get("logs") or {}).get("output_dir") or "logs"
        return (PROJECT_ROOT / rel).resolve()
    except Exception:
        return (PROJECT_ROOT / "logs").resolve()


def _paths():
    base = _logs_base()
    return {
        "base": base,
        "csv": base / "events.csv",
        "jsonl": base / "events.jsonl",
        "screenshots": base / "screenshots",
        "clips": base / "clips",
    }


def _under_logs(path: Path) -> bool:
    try:
        path.resolve().relative_to(_logs_base().resolve())
        return True
    except ValueError:
        return False


app = FastAPI(
    title="Drowning Monitor · 看板",
    version="0.2",
    description="只读本地日志；与桌面监测进程独立，未启动本服务不影响桌面端。",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/api/health")
def health():
    p = _paths()
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "logs_dir": str(p["base"]),
        "has_jsonl": p["jsonl"].is_file(),
        "has_csv": p["csv"].is_file(),
    }


@app.get("/api/events")
def api_events(limit: int = 100):
    limit = max(1, min(limit, 500))
    p = _paths()

    if p["jsonl"].is_file():
        try:
            lines = p["jsonl"].read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise HTTPException(500, detail=f"读取 jsonl 失败: {e}") from e
        out = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {"source": "jsonl", "items": out, "logs_dir": str(p["base"])}

    if not p["csv"].is_file():
        return {"source": "none", "items": [], "rows": [], "logs_dir": str(p["base"])}

    try:
        with open(p["csv"], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except OSError as e:
        raise HTTPException(500, detail=f"读取 csv 失败: {e}") from e

    if len(rows) <= 1:
        return {"source": "csv", "items": [], "header": rows[0] if rows else [], "rows": []}
    header, data = rows[0], rows[1:]
    tail = data[-limit:]
    return {
        "source": "csv",
        "header": header,
        "rows": tail,
        "logs_dir": str(p["base"]),
    }


@app.get("/api/stats")
def api_stats():
    p = _paths()
    alerts = 0
    clips = 0
    csv_rows = 0

    if p["jsonl"].is_file():
        try:
            for line in p["jsonl"].read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = o.get("kind")
                if k == "drowning_alert":
                    alerts += 1
                elif k == "clip_saved":
                    clips += 1
        except OSError:
            pass

    if p["csv"].is_file():
        try:
            with open(p["csv"], "r", encoding="utf-8") as f:
                csv_rows = max(0, sum(1 for _ in f) - 1)
        except OSError:
            csv_rows = 0

    return {
        "drowning_alert_events": alerts,
        "clip_saved_events": clips,
        "csv_data_rows": csv_rows,
        "logs_dir": str(p["base"]),
    }


@app.get("/api/stats/hourly")
def api_stats_hourly(hours: int = 24):
    """最近 N 小时内每小时 drowning_alert 次数（供图表）。"""
    hours = max(1, min(hours, 168))
    p = _paths()
    buckets: dict[str, int] = defaultdict(int)
    if not p["jsonl"].is_file():
        return {"hours": hours, "series": []}

    try:
        lines = p["jsonl"].read_text(encoding="utf-8").splitlines()
    except OSError:
        return {"hours": hours, "series": []}

    now = datetime.now()
    cutoff = now.timestamp() - hours * 3600

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if o.get("kind") != "drowning_alert":
            continue
        ts = o.get("time_local") or o.get("time_iso") or ""
        try:
            if "T" in str(o.get("time_iso", "")):
                dt = datetime.fromisoformat(str(o["time_iso"]).replace("Z", "+00:00"))
            else:
                dt = datetime.strptime(str(ts)[:19], "%Y-%m-%d %H:%M:%S")
        except (ValueError, TypeError, KeyError):
            continue
        if dt.timestamp() < cutoff:
            continue
        key = dt.strftime("%Y-%m-%d %H:00")
        buckets[key] += 1

    series = sorted([{"bucket": k, "count": v} for k, v in buckets.items()], key=lambda x: x["bucket"])
    return {"hours": hours, "series": series}


@app.get("/files/screenshots/{filename:path}")
def file_screenshot(filename: str):
    p = _paths()
    fp = (p["screenshots"] / filename).resolve()
    if not _under_logs(fp) or not fp.is_file():
        raise HTTPException(404)
    return FileResponse(fp)


@app.get("/files/clips/{filename:path}")
def file_clip(filename: str):
    p = _paths()
    fp = (p["clips"] / filename).resolve()
    if not _under_logs(fp) or not fp.is_file():
        raise HTTPException(404)
    return FileResponse(fp, media_type="video/x-msvideo")


@app.get("/")
def index():
    index_html = STATIC_DIR / "index.html"
    if index_html.is_file():
        return FileResponse(index_html)
    return {"message": "缺少 web/static/index.html", "docs": "/docs"}


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")
