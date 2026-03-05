#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bisignlangtrans.data.features import build_sequence_features
from bisignlangtrans.data.keypoints import (
    extract_frame_keypoints,
    extract_video_keypoints,
    resolve_mediapipe_apis,
)
from bisignlangtrans.decoding import ctc_greedy_decode, ids_to_tokens
from bisignlangtrans.models import BiLSTMCTC


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model_from_ckpt(ckpt: dict, device: torch.device):
    model_cfg = ckpt["config"]
    input_dim = int(model_cfg["input_dim"])
    num_classes = int(model_cfg["num_classes"])
    model = BiLSTMCTC(
        input_dim=input_dim,
        num_classes=num_classes,
        proj_dim=int(model_cfg.get("proj_dim", 192)),
        hidden_size=int(model_cfg.get("hidden_size", 192)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_tokens_from_keypoints(
    keypoints: np.ndarray,
    model,
    device: torch.device,
    blank_id: int,
    id_to_token,
    normalize: bool,
    use_velocity: bool,
) -> tuple[List[int], List[str]]:
    feats = build_sequence_features(
        keypoints,
        normalize=normalize,
        use_velocity=use_velocity,
        flatten=True,
    )
    x = torch.from_numpy(feats).unsqueeze(0).to(device=device, dtype=torch.float32)
    lengths = torch.tensor([x.shape[1]], dtype=torch.long, device=device)
    logits = model(x, lengths)
    pred_ids = ctc_greedy_decode(logits, lengths, blank_id=blank_id)[0]
    pred_tokens = ids_to_tokens(pred_ids, id_to_token)
    return pred_ids, pred_tokens


def _draw_overlay(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    line_h = 28
    margin = 12
    box_h = margin * 2 + line_h * len(lines)
    cv2.rectangle(out, (0, 0), (w, min(box_h, h)), (0, 0, 0), thickness=-1)
    y = margin + 20
    for ln in lines:
        cv2.putText(
            out,
            ln,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += line_h
    return out


def _short_tokens_text(tokens: List[str], max_chars: int = 80) -> str:
    if not tokens:
        return "<EMPTY>"
    s = "/".join(tokens)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def run_camera_mode(
    camera_id: int,
    model,
    device: torch.device,
    blank_id: int,
    id_to_token,
    normalize: bool,
    use_velocity: bool,
    frame_stride: int,
    max_frames: int,
    infer_every: int,
    min_frames: int,
    window_frames: int,
) -> dict:
    if frame_stride <= 0:
        raise ValueError(f"--frame-stride must be > 0, got {frame_stride}")
    if infer_every <= 0:
        raise ValueError(f"--camera-infer-every must be > 0, got {infer_every}")
    if min_frames <= 0:
        raise ValueError(f"--camera-min-frames must be > 0, got {min_frames}")
    if window_frames < 0:
        raise ValueError(f"--camera-window-frames must be >= 0, got {window_frames}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: id={camera_id}")

    try:
        cv2.namedWindow("K2G-BiCTC Camera", cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        cap.release()
        raise RuntimeError("Unable to create OpenCV window. Camera mode requires GUI support.") from exc

    mp_hands, mp_pose = resolve_mediapipe_apis()
    cv2.setNumThreads(0)

    keypoint_buffer: List[np.ndarray] = []
    pred_ids: List[int] = []
    pred_tokens: List[str] = []
    frame_idx = 0
    infer_tick = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_stride == 0:
                frame_kpts = extract_frame_keypoints(frame, hands=hands, pose=pose)
                keypoint_buffer.append(frame_kpts)

                if max_frames > 0 and len(keypoint_buffer) > max_frames:
                    keypoint_buffer = keypoint_buffer[-max_frames:]

                if len(keypoint_buffer) >= min_frames:
                    infer_tick += 1
                    if infer_tick % infer_every == 0:
                        use = keypoint_buffer[-window_frames:] if window_frames > 0 else keypoint_buffer
                        clip_kpts = np.stack(use, axis=0)
                        pred_ids, pred_tokens = predict_tokens_from_keypoints(
                            keypoints=clip_kpts,
                            model=model,
                            device=device,
                            blank_id=blank_id,
                            id_to_token=id_to_token,
                            normalize=normalize,
                            use_velocity=use_velocity,
                        )

            text = _short_tokens_text(pred_tokens, max_chars=72)
            overlay = _draw_overlay(
                frame,
                [
                    "q/Esc: quit  r: reset",
                    f"frames={len(keypoint_buffer)}  stride={frame_stride}",
                    "decode=greedy",
                    f"pred={text}",
                ],
            )
            cv2.imshow("K2G-BiCTC Camera", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                keypoint_buffer.clear()
                pred_ids = []
                pred_tokens = []
                infer_tick = 0

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return {
        "camera_id": camera_id,
        "frames": len(keypoint_buffer),
        "decode_method": "greedy",
        "pred_token_ids": pred_ids,
        "pred_tokens": pred_tokens,
        "pred_joined": " ".join(pred_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline inference for BiLSTM+CTC model")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, default="")
    input_group.add_argument("--camera-id", type=int, default=-1)
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--camera-infer-every", type=int, default=6)
    parser.add_argument("--camera-min-frames", type=int, default=24)
    parser.add_argument("--camera-window-frames", type=int, default=96)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    video_path: Optional[Path] = None
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
    elif args.camera_id < 0:
        raise ValueError("Either --video or non-negative --camera-id is required")

    device = pick_device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model_from_ckpt(ckpt=ckpt, device=device)
    vocab = ckpt["vocab"]
    blank_id = int(ckpt["blank_id"])
    id_to_token = vocab["id_to_token"]
    normalize = bool(ckpt["config"].get("normalize", True))
    use_velocity = bool(ckpt["config"].get("use_velocity", True))

    if video_path is not None:
        mp_hands, mp_pose = resolve_mediapipe_apis()
        cv2.setNumThreads(0)
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands, mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            keypoints, fps = extract_video_keypoints(
                video_path=video_path,
                hands=hands,
                pose=pose,
                frame_stride=int(args.frame_stride),
                max_frames=int(args.max_frames),
            )

        pred_ids, pred_tokens = predict_tokens_from_keypoints(
            keypoints=keypoints,
            model=model,
            device=device,
            blank_id=blank_id,
            id_to_token=id_to_token,
            normalize=normalize,
            use_velocity=use_velocity,
        )

        payload = {
            "video_path": str(video_path.resolve()),
            "decode_method": "greedy",
            "frames": int(keypoints.shape[0]),
            "fps": float(fps),
            "pred_token_ids": pred_ids,
            "pred_tokens": pred_tokens,
            "pred_joined": " ".join(pred_tokens),
        }
        print(f"[infer] video={video_path.resolve()}")
        print(f"[infer] frames={keypoints.shape[0]} fps={fps:.3f}")
        print(f"[infer] decode=greedy pred_tokens={'/'.join(pred_tokens)}")
    else:
        payload = run_camera_mode(
            camera_id=int(args.camera_id),
            model=model,
            device=device,
            blank_id=blank_id,
            id_to_token=id_to_token,
            normalize=normalize,
            use_velocity=use_velocity,
            frame_stride=int(args.frame_stride),
            max_frames=int(args.max_frames),
            infer_every=int(args.camera_infer_every),
            min_frames=int(args.camera_min_frames),
            window_frames=int(args.camera_window_frames),
        )
        print(f"[infer] camera_id={payload['camera_id']} frames={payload['frames']}")
        print(f"[infer] decode=greedy pred_tokens={'/'.join(payload['pred_tokens'])}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[infer] saved: {out.resolve()}")


if __name__ == "__main__":
    main()
