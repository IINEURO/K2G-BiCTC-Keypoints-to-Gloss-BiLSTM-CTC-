from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

POSE_UPPER_BODY_INDICES: List[int] = [
    0,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
]


def resolve_mediapipe_apis():
    try:
        return mp.solutions.hands, mp.solutions.pose
    except Exception:
        pass

    try:
        from mediapipe.python.solutions import hands as mp_hands  # type: ignore
        from mediapipe.python.solutions import pose as mp_pose  # type: ignore

        return mp_hands, mp_pose
    except Exception as exc:
        mp_ver = getattr(mp, "__version__", "unknown")
        raise RuntimeError(
            "Cannot access MediaPipe Hands/Pose APIs. "
            f"Detected mediapipe={mp_ver}. This project requires the legacy solutions API.\n"
            "Fix with:\n"
            "  pip uninstall -y mediapipe mediapipe-nightly\n"
            "  pip install mediapipe==0.10.14"
        ) from exc


def extract_frame_keypoints(image_bgr: np.ndarray, hands, pose) -> np.ndarray:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid frame for keypoint extraction")

    frame_kpts = np.zeros((55, 4), dtype=np.float32)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    hands_res = hands.process(image_rgb)
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_lms, handedness in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
            label = handedness.classification[0].label.lower()
            offset = 0 if label == "left" else 21
            for i, lm in enumerate(hand_lms.landmark):
                frame_kpts[offset + i] = [lm.x, lm.y, lm.z, 1.0]

    pose_res = pose.process(image_rgb)
    if pose_res.pose_landmarks:
        for local_i, pose_i in enumerate(POSE_UPPER_BODY_INDICES):
            lm = pose_res.pose_landmarks.landmark[pose_i]
            frame_kpts[42 + local_i] = [lm.x, lm.y, lm.z, float(lm.visibility)]

    return frame_kpts


def extract_video_keypoints(
    video_path: str | Path,
    hands,
    pose,
    frame_stride: int = 1,
    max_frames: int = 0,
) -> Tuple[np.ndarray, float]:
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0, got {frame_stride}")
    if max_frames < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames}")

    p = Path(video_path)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {p}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    out: List[np.ndarray] = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            out.append(extract_frame_keypoints(frame, hands, pose))
            if max_frames > 0 and len(out) >= max_frames:
                break
        idx += 1

    cap.release()

    if not out:
        raise RuntimeError(f"No frames decoded from video: {p}")

    return np.stack(out, axis=0), fps
