"""加载 config/settings.json；不存在则用内置默认（与 settings.example.json 一致）"""

import json
import os
from pathlib import Path

from utils.paths import PROJECT_ROOT

_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"
_EXAMPLE_PATH = PROJECT_ROOT / "config" / "settings.example.json"

_DEFAULTS = {
    "classifier": {"path": "weights/classifier_best.pth"},
    "deepseek": {
        "api_key": "",
        "api_base": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "logs": {"output_dir": "logs"},
    "dingtalk": {
        "enabled": False,
        "webhook": "",
        "secret": "",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings() -> dict:
    data = dict(_DEFAULTS)
    if _CONFIG_PATH.is_file():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            user = json.load(f)
        data = _deep_merge(data, user)
    elif _EXAMPLE_PATH.is_file():
        try:
            with open(_EXAMPLE_PATH, "r", encoding="utf-8") as f:
                ex = json.load(f)
            data = _deep_merge(data, {k: v for k, v in ex.items() if not k.startswith("_")})
        except (json.JSONDecodeError, OSError):
            pass
    return data


_settings_cache: dict | None = None


def get_settings() -> dict:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_settings()
    return _settings_cache


def reload_settings() -> dict:
    global _settings_cache
    _settings_cache = None
    return get_settings()


def resolve_classifier_path(cli_path: str | None) -> str | None:
    """命令行优先，否则 config 中的相对路径转为绝对路径。"""
    if cli_path:
        p = Path(cli_path)
        return str(p.resolve()) if p.exists() else str(p)

    s = get_settings()
    rel = (s.get("classifier") or {}).get("path") or ""
    if not rel or not str(rel).strip():
        return None
    p = PROJECT_ROOT / rel
    return str(p.resolve()) if p.is_file() else None
