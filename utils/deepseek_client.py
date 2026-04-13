"""
DeepSeek API（OpenAI 兼容）— 预留：在 config/settings.json 填写 deepseek.api_key 后使用。

用途：将近期报警摘要转为自然语言简报，不参与溺水判定。
"""

from __future__ import annotations

import json
from typing import Any

import requests

from utils.settings import get_settings


def is_configured() -> bool:
    key = (get_settings().get("deepseek") or {}).get("api_key") or ""
    return bool(str(key).strip())


def _endpoint() -> str:
    s = get_settings().get("deepseek") or {}
    base = (s.get("api_base") or "https://api.deepseek.com").rstrip("/")
    return f"{base}/v1/chat/completions"


def _headers() -> dict[str, str]:
    key = (get_settings().get("deepseek") or {}).get("api_key") or ""
    return {
        "Authorization": f"Bearer {key.strip()}",
        "Content-Type": "application/json",
    }


def summarize_events_text(user_payload: str, system_prompt: str | None = None) -> str:
    """
    发送一段已整理好的中文文本（如最近 CSV/日志摘要），返回模型生成的简报。
    """
    if not is_configured():
        raise RuntimeError("未配置 DeepSeek：请在 config/settings.json 中填写 deepseek.api_key")

    s = get_settings().get("deepseek") or {}
    model = s.get("model") or "deepseek-chat"
    sys_msg = system_prompt or (
        "你是泳池与野外水域安全监测系统的助手。根据用户提供的报警与统计摘要，"
        "用简洁、专业的中文写一段值班简报（含时间范围、事件条数、建议关注项）。"
        "不要编造数据中不存在的事实。"
    )

    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_payload},
        ],
        "temperature": 0.3,
    }

    r = requests.post(_endpoint(), headers=_headers(), json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def build_payload_from_recent_events(rows: list[list[str]], max_lines: int = 30) -> str:
    """将 get_recent_events 返回的表格行拼成发给模型的文本。"""
    if not rows:
        return "（当前无报警记录）"
    lines = []
    for row in rows[:max_lines]:
        lines.append(" | ".join(row))
    return "最近事件（表格式列）：\n" + "\n".join(lines)
