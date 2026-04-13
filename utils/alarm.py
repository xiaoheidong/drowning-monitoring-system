"""声音报警模块 - 检测到溺水时发出声音警报"""

import threading
import winsound
import time


class AlarmManager:
    """管理报警声音，防止重复触发"""

    def __init__(self, cooldown_seconds=5):
        self.cooldown = cooldown_seconds
        self._last_alarm_time = 0
        self._is_playing = False
        self._enabled = True
        self._alarm_thread = None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value

    def trigger(self):
        """触发报警（带冷却时间，防止连续鸣叫）"""
        if not self._enabled:
            return
        now = time.time()
        if now - self._last_alarm_time < self.cooldown:
            return
        if self._is_playing:
            return

        self._last_alarm_time = now
        self._alarm_thread = threading.Thread(target=self._play_alarm, daemon=True)
        self._alarm_thread.start()

    def _play_alarm(self):
        self._is_playing = True
        try:
            for freq in [1200, 800, 1200, 800, 1200]:
                if not self._enabled:
                    break
                winsound.Beep(freq, 300)
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            self._is_playing = False

    def stop(self):
        self._enabled = False
        self._is_playing = False
