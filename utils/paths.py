"""项目根目录（供绝对路径、配置文件、Web 读盘使用）"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
