"""钉钉群机器人 Webhook 推送（支持加签 SEC）。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from utils.settings import get_settings


def _signed_webhook_url(webhook: str, secret: str) -> str:
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(
        secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(digest))
    sep = "&" if "?" in webhook else "?"
    return f"{webhook}{sep}timestamp={timestamp}&sign={sign}"


def send_dingtalk_markdown(webhook: str, secret: str | None, title: str, text: str) -> tuple[bool, str]:
    """POST markdown 消息；secret 为空则不加签（仅适用于未开加签的机器人）。"""
    url = _signed_webhook_url(webhook, secret) if (secret or "").strip() else webhook
    body = {
        "msgtype": "markdown",
        "markdown": {"title": title, "text": text},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return False, raw
    except OSError as e:
        return False, str(e)

    try:
        o = json.loads(raw)
    except json.JSONDecodeError:
        return False, raw

    errcode = o.get("errcode", -1)
    if errcode == 0:
        return True, raw
    return False, o.get("errmsg", raw)


def maybe_send_dingtalk_alert(
    *,
    time_str: str,
    person_count: int,
    drowning_count: int,
    confidence: float,
    note: str,
) -> None:
    s = (get_settings().get("dingtalk") or {})
    if not s.get("enabled"):
        return
    webhook = (s.get("webhook") or "").strip()
    if not webhook:
        return
    secret = (s.get("secret") or "").strip() or None

    title = "溺水监测告警"
    text = (
        f"### {title}\n\n"
        f"- **时间**：{time_str}\n"
        f"- **现场人数**：{person_count}\n"
        f"- **确认溺水人数**：{drowning_count}\n"
        f"- **置信度**：{confidence:.1%}\n"
        f"- **备注**：{note or '—'}\n"
    )
    ok, info = send_dingtalk_markdown(webhook, secret, title, text)
    if not ok:
        print(f"[钉钉] 推送失败: {info}")
