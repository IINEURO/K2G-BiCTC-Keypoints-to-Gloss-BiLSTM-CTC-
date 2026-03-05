#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mproc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bisignlangtrans.data.ce_csl import load_manifest, npz_path_for_video
from bisignlangtrans.data.keypoints import extract_video_keypoints, resolve_mediapipe_apis


def _process_rows(
    row_items: List[Tuple[str, str, str, str]],
    processed_root: str,
    overwrite: bool,
    frame_stride: int,
    max_frames: int,
    worker_name: str,
) -> Tuple[int, int, int]:
    cv2.setNumThreads(0)
    mp_hands, mp_pose = resolve_mediapipe_apis()

    ok = 0
    skip = 0
    done = 0

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
        for video_path, split, number, translator in row_items:
            done += 1
            try:
                out = npz_path_for_video(processed_root, split, video_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                if out.exists() and not overwrite:
                    skip += 1
                    continue

                keypoints, fps = extract_video_keypoints(
                    video_path=video_path,
                    hands=hands,
                    pose=pose,
                    frame_stride=frame_stride,
                    max_frames=max_frames,
                )
                np.savez_compressed(
                    out,
                    keypoints=keypoints,
                    fps=np.float32(fps),
                    num_frames=np.int32(keypoints.shape[0]),
                    split=np.array(split),
                    number=np.array(number),
                    translator=np.array(translator),
                    video_path=np.array(video_path),
                )
                ok += 1
            except Exception as exc:
                raise RuntimeError(
                    f"[{worker_name}] failed: split={split} number={number} "
                    f"translator={translator} video={video_path}"
                ) from exc

    return ok, skip, done


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CE-CSL keypoints to npz")
    parser.add_argument("--manifest", type=str, default="data/meta/manifest.jsonl")
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--split", type=str, default="", choices=["", "train", "dev", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError(f"--frame-stride must be > 0, got {args.frame_stride}")
    if args.max_frames < 0:
        raise ValueError(f"--max-frames must be >= 0, got {args.max_frames}")
    if args.num_workers <= 0:
        raise ValueError(f"--num-workers must be > 0, got {args.num_workers}")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be > 0, got {args.chunk_size}")

    rows = load_manifest(args.manifest, split=args.split or None)
    if not rows:
        raise RuntimeError("No rows loaded from manifest")
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    print(
        f"[extract] manifest={Path(args.manifest).resolve()} rows={len(rows)} "
        f"processed_root={Path(args.processed_root).resolve()} split={args.split or 'all'} "
        f"workers={args.num_workers}"
    )

    row_items: List[Tuple[str, str, str, str]] = [
        (r.video_path, r.split, r.number, r.translator) for r in rows
    ]

    ok = 0
    skip = 0

    if args.num_workers == 1:
        ok, skip, _ = _process_rows(
            row_items=row_items,
            processed_root=args.processed_root,
            overwrite=args.overwrite,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            worker_name="w0",
        )
    else:
        chunks = [
            row_items[i : i + args.chunk_size]
            for i in range(0, len(row_items), args.chunk_size)
        ]
        print(
            f"[extract] multiprocessing enabled: workers={args.num_workers}, "
            f"chunks={len(chunks)}, chunk_size={args.chunk_size}"
        )
        ctx = mproc.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as ex:
            futures = [
                ex.submit(
                    _process_rows,
                    chunk,
                    args.processed_root,
                    args.overwrite,
                    args.frame_stride,
                    args.max_frames,
                    f"w{i}",
                )
                for i, chunk in enumerate(chunks)
            ]
            with tqdm(total=len(row_items), desc="extract") as pbar:
                for fut in as_completed(futures):
                    ok_i, skip_i, done_i = fut.result()
                    ok += ok_i
                    skip += skip_i
                    pbar.update(done_i)

    print(f"[extract] done: saved={ok}, skipped={skip}")


if __name__ == "__main__":
    main()
