"""
独立启动看板服务（与桌面端无关）。

    python -m web

若未安装依赖，仅本命令失败，不影响 python main.py。
"""

import sys


def main():
    try:
        import uvicorn
    except ImportError:
        print("[错误] 未安装 uvicorn。请执行: pip install fastapi uvicorn")
        sys.exit(1)
    uvicorn.run(
        "web.server:app",
        host="127.0.0.1",
        port=8080,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
