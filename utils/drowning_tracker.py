"""溺水时序判定追踪器 - 连续多帧确认后才触发报警，减少误报"""

import time
from collections import defaultdict


class DrowningTracker:
    """
    追踪每个检测目标的溺水状态。
    只有连续 confirm_frames 帧判定为溺水，才触发真正的报警。
    使用 IOU 做简单的跨帧目标关联。
    """

    def __init__(self, confirm_frames=8, timeout_seconds=3.0, iou_threshold=0.3):
        self.confirm_frames = confirm_frames
        self.timeout = timeout_seconds
        self.iou_threshold = iou_threshold

        self._tracks = {}
        self._next_id = 0

    def update(self, detections: list[dict], predictions: list[tuple]) -> list[dict]:
        """
        更新追踪状态。
        detections: [{"bbox": (x1,y1,x2,y2), "confidence": float}, ...]
        predictions: [(class_name, conf), ...]

        返回增强后的检测列表，包含:
            - confirmed_drowning: bool (是否确认溺水)
            - track_id: int
            - drowning_count: int (连续溺水帧数)
        """
        now = time.time()
        self._cleanup_stale_tracks(now)

        current_bboxes = [d["bbox"] for d in detections]
        matched, unmatched_dets = self._match_tracks(current_bboxes)

        results = [None] * len(detections)

        for det_idx, track_id in matched.items():
            class_name = predictions[det_idx][0] if det_idx < len(predictions) else "unknown"
            track = self._tracks[track_id]
            track["last_seen"] = now
            track["bbox"] = current_bboxes[det_idx]

            if class_name == "drowning":
                track["drowning_count"] += 1
            else:
                track["drowning_count"] = max(0, track["drowning_count"] - 2)

            confirmed = track["drowning_count"] >= self.confirm_frames

            results[det_idx] = {
                **detections[det_idx],
                "track_id": track_id,
                "confirmed_drowning": confirmed,
                "drowning_count": track["drowning_count"],
            }

        for det_idx in unmatched_dets:
            track_id = self._next_id
            self._next_id += 1
            class_name = predictions[det_idx][0] if det_idx < len(predictions) else "unknown"

            self._tracks[track_id] = {
                "bbox": current_bboxes[det_idx],
                "last_seen": now,
                "drowning_count": 1 if class_name == "drowning" else 0,
            }

            results[det_idx] = {
                **detections[det_idx],
                "track_id": track_id,
                "confirmed_drowning": False,
                "drowning_count": self._tracks[track_id]["drowning_count"],
            }

        return [r for r in results if r is not None]

    def _match_tracks(self, bboxes):
        """用 IOU 做贪心匹配"""
        matched = {}
        unmatched_dets = set(range(len(bboxes)))

        if not self._tracks or not bboxes:
            return matched, unmatched_dets

        track_ids = list(self._tracks.keys())
        track_bboxes = [self._tracks[tid]["bbox"] for tid in track_ids]

        iou_matrix = []
        for det_bbox in bboxes:
            row = [self._compute_iou(det_bbox, tb) for tb in track_bboxes]
            iou_matrix.append(row)

        used_tracks = set()
        pairs = []
        for i in range(len(bboxes)):
            for j in range(len(track_ids)):
                if iou_matrix[i][j] >= self.iou_threshold:
                    pairs.append((iou_matrix[i][j], i, j))

        pairs.sort(reverse=True)

        for iou_val, det_idx, trk_idx in pairs:
            if det_idx in matched or trk_idx in used_tracks:
                continue
            matched[det_idx] = track_ids[trk_idx]
            used_tracks.add(trk_idx)
            unmatched_dets.discard(det_idx)

        return matched, unmatched_dets

    @staticmethod
    def _compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0

    def _cleanup_stale_tracks(self, now):
        stale = [tid for tid, t in self._tracks.items()
                 if now - t["last_seen"] > self.timeout]
        for tid in stale:
            del self._tracks[tid]

    def reset(self):
        self._tracks.clear()
        self._next_id = 0

    @property
    def active_drowning_count(self):
        return sum(1 for t in self._tracks.values()
                   if t["drowning_count"] >= self.confirm_frames)
