from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import build_sequence_features

PUNCT_TOKENS = {"。", "？", "?", "，", ",", "！", "!", "；", ";"}


@dataclass
class CECSLManifestRow:
    split: str
    number: str
    translator: str
    video_path: str
    chinese_sentence: str
    gloss_tokens: List[str]


def gloss_to_tokens(gloss: str, drop_punct: bool = True) -> List[str]:
    raw = [t.strip() for t in gloss.split("/") if t.strip()]
    if not drop_punct:
        return raw
    return [t for t in raw if t not in PUNCT_TOKENS]


def _iter_rows_from_label_csv(
    label_csv_path: Path,
    split: str,
    video_root: Path,
    drop_punct: bool,
) -> Iterable[CECSLManifestRow]:
    with label_csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Number", "Translator", "Chinese Sentences", "Gloss"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {label_csv_path}: {sorted(missing)}")

        for row in reader:
            number = row["Number"].strip()
            translator = row["Translator"].strip()
            sentence = row["Chinese Sentences"].strip()
            gloss = row["Gloss"].strip()
            tokens = gloss_to_tokens(gloss, drop_punct=drop_punct)
            if not tokens:
                continue

            video_path = video_root / split / translator / f"{number}.mp4"
            yield CECSLManifestRow(
                split=split,
                number=number,
                translator=translator,
                video_path=str(video_path),
                chinese_sentence=sentence,
                gloss_tokens=tokens,
            )


def write_manifest_and_vocab(
    raw_root: str | Path,
    manifest_path: str | Path,
    vocab_path: str | Path,
    min_freq: int = 1,
    drop_punct: bool = True,
    use_unk: bool = True,
) -> None:
    raw_root = Path(raw_root)
    label_dir = raw_root / "label"
    video_root = raw_root / "video"

    if not label_dir.exists():
        raise FileNotFoundError(f"label dir not found: {label_dir}")
    if not video_root.exists():
        raise FileNotFoundError(f"video dir not found: {video_root}")

    rows: List[CECSLManifestRow] = []
    token_freq = Counter()

    for split in ["train", "dev", "test"]:
        label_csv = label_dir / f"{split}.csv"
        if not label_csv.exists():
            raise FileNotFoundError(f"Missing split label file: {label_csv}")

        for item in _iter_rows_from_label_csv(
            label_csv_path=label_csv,
            split=split,
            video_root=video_root,
            drop_punct=drop_punct,
        ):
            if not Path(item.video_path).exists():
                raise FileNotFoundError(f"Missing video file referenced by label: {item.video_path}")
            rows.append(item)
            token_freq.update(item.gloss_tokens)

    keep_tokens = sorted([t for t, c in token_freq.items() if c >= min_freq])

    token_to_id: Dict[str, int] = {"<BLANK>": 0}
    if use_unk:
        token_to_id["<UNK>"] = len(token_to_id)
    for t in keep_tokens:
        if t in token_to_id:
            continue
        token_to_id[t] = len(token_to_id)

    id_to_token = [None] * len(token_to_id)
    for t, i in token_to_id.items():
        id_to_token[i] = t

    vocab = {
        "blank_token": "<BLANK>",
        "blank_id": 0,
        "unk_token": "<UNK>" if use_unk else None,
        "unk_id": int(token_to_id["<UNK>"]) if use_unk else None,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "size": len(token_to_id),
        "min_freq": min_freq,
        "drop_punct": drop_punct,
        "use_unk": use_unk,
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "split": r.split,
                        "number": r.number,
                        "translator": r.translator,
                        "video_path": r.video_path,
                        "chinese_sentence": r.chinese_sentence,
                        "gloss_tokens": r.gloss_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    vocab_path = Path(vocab_path)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_manifest(manifest_path: str | Path, split: str | None = None) -> List[CECSLManifestRow]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out: List[CECSLManifestRow] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                row = CECSLManifestRow(
                    split=obj["split"],
                    number=obj["number"],
                    translator=obj["translator"],
                    video_path=obj["video_path"],
                    chinese_sentence=obj["chinese_sentence"],
                    gloss_tokens=list(obj["gloss_tokens"]),
                )
                if split is None or row.split == split:
                    out.append(row)
            except Exception as exc:
                raise ValueError(f"Bad manifest row at line {ln}") from exc
    return out


def load_vocab(vocab_path: str | Path) -> dict:
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")
    with vocab_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "token_to_id" not in obj or "blank_id" not in obj:
        raise ValueError("Vocab file missing required keys: token_to_id/blank_id")
    return obj


def npz_path_for_video(processed_root: str | Path, split: str, video_path: str) -> Path:
    key = hashlib.sha1(video_path.encode("utf-8")).hexdigest()[:16]
    return Path(processed_root) / split / f"{key}.npz"


def tokens_to_ids(tokens: Sequence[str], token_to_id: Dict[str, int]) -> List[int]:
    ids = []
    unk_id = token_to_id.get("<UNK>")
    for t in tokens:
        if t not in token_to_id:
            if unk_id is None:
                raise KeyError(f"Token not in vocabulary: {t}")
            ids.append(int(unk_id))
            continue
        ids.append(int(token_to_id[t]))
    return ids


class CTCKeypointDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        vocab_path: str | Path,
        processed_root: str | Path,
        split: str,
        normalize: bool = True,
        use_velocity: bool = True,
        strict: bool = True,
    ) -> None:
        self.rows = load_manifest(manifest_path, split=split)
        if not self.rows:
            raise ValueError(f"No rows for split={split}")

        vocab = load_vocab(vocab_path)
        self.token_to_id: Dict[str, int] = {k: int(v) for k, v in vocab["token_to_id"].items()}
        self.processed_root = Path(processed_root)
        self.normalize = normalize
        self.use_velocity = use_velocity

        self.samples: List[tuple[CECSLManifestRow, Path, List[int]]] = []
        missing = []
        for r in self.rows:
            p = npz_path_for_video(self.processed_root, r.split, r.video_path)
            if not p.exists():
                missing.append(p)
                continue
            y = tokens_to_ids(r.gloss_tokens, self.token_to_id)
            self.samples.append((r, p, y))

        if strict and missing:
            preview = "\n".join(str(p) for p in missing[:5])
            raise FileNotFoundError(
                "Missing processed keypoint npz files. Run scripts/extract_keypoints.py first. "
                f"Examples:\n{preview}"
            )

        if not self.samples:
            raise RuntimeError(f"No usable samples for split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row, npz_path, y = self.samples[idx]
        arr = np.load(npz_path)
        if "keypoints" not in arr:
            raise KeyError(f"npz missing keypoints: {npz_path}")

        keypoints = arr["keypoints"].astype(np.float32)
        if keypoints.ndim != 3 or keypoints.shape[1:] != (55, 4):
            raise ValueError(f"Unexpected keypoint shape in {npz_path}: {keypoints.shape}")

        x = build_sequence_features(
            keypoints,
            normalize=self.normalize,
            use_velocity=self.use_velocity,
            flatten=True,
        )

        x_t = torch.from_numpy(x)
        y_t = torch.tensor(y, dtype=torch.long)
        meta = {
            "number": row.number,
            "translator": row.translator,
            "video_path": row.video_path,
            "gloss_tokens": row.gloss_tokens,
        }
        return x_t, y_t, meta


def ctc_collate(batch):
    xs, ys, metas = zip(*batch)

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    target_lengths = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)

    feat_dim = xs[0].shape[1]
    max_t = int(lengths.max().item())
    bsz = len(xs)

    x_pad = torch.zeros((bsz, max_t, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0]] = x

    targets = torch.cat(ys, dim=0)
    return x_pad, lengths, targets, target_lengths, list(metas)
