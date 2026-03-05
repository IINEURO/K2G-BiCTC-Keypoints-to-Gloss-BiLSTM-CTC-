#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bisignlangtrans.data.ce_csl import CTCKeypointDataset, ctc_collate, load_vocab
from bisignlangtrans.decoding import ctc_greedy_decode
from bisignlangtrans.models import BiLSTMCTC


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def split_targets(targets: torch.Tensor, target_lengths: torch.Tensor) -> List[List[int]]:
    out: List[List[int]] = []
    idx = 0
    for l in target_lengths.tolist():
        l = int(l)
        out.append(targets[idx : idx + l].tolist())
        idx += l
    return out


def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, len(b) + 1):
            old = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1]


def build_scheduler(optimizer, scheduler_name: str, epochs: int, min_lr: float):
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x_pad, input_lengths, targets, target_lengths, _ in pbar:
        x_pad = x_pad.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_pad, input_lengths)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bsz = x_pad.shape[0]
        total_loss += float(loss.item()) * bsz
        total_samples += bsz
        pbar.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    ctc_loss_fn: nn.Module,
    device: torch.device,
    blank_id: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    total_seq = 0
    seq_match = 0
    total_edit = 0
    total_target_tokens = 0

    pbar = tqdm(loader, desc="eval", leave=False)
    for x_pad, input_lengths, targets, target_lengths, _ in pbar:
        x_pad = x_pad.to(device, non_blocking=True)
        input_lengths = input_lengths.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(x_pad, input_lengths)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        bsz = x_pad.shape[0]
        total_loss += float(loss.item()) * bsz
        total_samples += bsz

        pred_ids = ctc_greedy_decode(logits, input_lengths, blank_id=blank_id)
        gt_ids = split_targets(targets.detach().cpu(), target_lengths.detach().cpu())

        for p, g in zip(pred_ids, gt_ids):
            total_seq += 1
            if p == g:
                seq_match += 1
            total_edit += edit_distance(p, g)
            total_target_tokens += len(g)

    return {
        "loss": total_loss / max(total_samples, 1),
        "seq_acc": seq_match / max(total_seq, 1),
        "ter": total_edit / max(total_target_tokens, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiLSTM+CTC on CE-CSL keypoints")
    parser.add_argument("--config", type=str, default="configs/bilstm_ctc.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    out_cfg = cfg["output"]

    model_name = str(model_cfg.get("name", "bilstm_ctc")).strip().lower()
    if model_name not in {"bilstm", "bilstm_ctc"}:
        raise ValueError("This script only supports model.name = bilstm_ctc")

    device = pick_device(str(train_cfg.get("device", "auto")))
    print(f"[train] device={device}")

    vocab = load_vocab(data_cfg["vocab_path"])
    blank_id = int(vocab["blank_id"])
    num_classes = int(vocab["size"])

    train_ds = CTCKeypointDataset(
        manifest_path=data_cfg["manifest_path"],
        vocab_path=data_cfg["vocab_path"],
        processed_root=data_cfg["processed_root"],
        split="train",
        normalize=bool(data_cfg.get("normalize", True)),
        use_velocity=bool(data_cfg.get("use_velocity", True)),
        strict=True,
    )
    dev_ds = CTCKeypointDataset(
        manifest_path=data_cfg["manifest_path"],
        vocab_path=data_cfg["vocab_path"],
        processed_root=data_cfg["processed_root"],
        split="dev",
        normalize=bool(data_cfg.get("normalize", True)),
        use_velocity=bool(data_cfg.get("use_velocity", True)),
        strict=True,
    )

    sample_x, _, _ = train_ds[0]
    input_dim = int(sample_x.shape[-1])

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=ctc_collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=ctc_collate,
    )

    model = BiLSTMCTC(
        input_dim=input_dim,
        num_classes=num_classes,
        proj_dim=int(model_cfg.get("proj_dim", 192)),
        hidden_size=int(model_cfg.get("hidden_size", 192)),
        num_layers=int(model_cfg.get("num_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)

    ctc_loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=str(train_cfg.get("scheduler", "cosine")),
        epochs=int(train_cfg["epochs"]),
        min_lr=float(train_cfg.get("min_lr", 1e-5)),
    )

    ckpt_dir = Path(out_cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    best_metric = str(train_cfg.get("best_metric", "seq_acc")).strip().lower()
    if best_metric not in {"seq_acc", "ter"}:
        raise ValueError(f"Unsupported best_metric: {best_metric}. Use 'seq_acc' or 'ter'.")
    best_metric_value = -1.0 if best_metric == "seq_acc" else float("inf")
    best_dev_seq_acc = -1.0
    best_dev_ter = float("inf")
    best_epoch = 0
    early_stop_patience = int(train_cfg.get("early_stop_patience", 0))
    early_stop_min_delta = float(train_cfg.get("early_stop_min_delta", 0.0))
    no_improve_epochs = 0
    final_epoch = 0

    print(
        f"[train] train={len(train_ds)} dev={len(dev_ds)} input_dim={input_dim} "
        f"num_classes={num_classes} blank_id={blank_id} model=bilstm_ctc "
        f"best_metric={best_metric} early_stop_patience={early_stop_patience}"
    )

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        final_epoch = epoch
        lr_now = optimizer.param_groups[0]["lr"]

        train_loss = run_train_epoch(
            model=model,
            loader=train_loader,
            ctc_loss_fn=ctc_loss_fn,
            optimizer=optimizer,
            device=device,
            grad_clip=float(train_cfg.get("grad_clip", 0.0)),
        )
        dv = run_eval_epoch(
            model=model,
            loader=dev_loader,
            ctc_loss_fn=ctc_loss_fn,
            device=device,
            blank_id=blank_id,
        )

        print(
            f"[epoch {epoch:03d}] lr={lr_now:.8f} "
            f"train_loss={train_loss:.4f} "
            f"dev_loss={dv['loss']:.4f} dev_seq_acc={dv['seq_acc']:.4f} dev_TER={dv['ter']:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

        curr_metric = float(dv["seq_acc"]) if best_metric == "seq_acc" else float(dv["ter"])
        is_better = (
            curr_metric > best_metric_value + early_stop_min_delta
            if best_metric == "seq_acc"
            else curr_metric < best_metric_value - early_stop_min_delta
        )
        if is_better:
            best_metric_value = curr_metric
            best_dev_seq_acc = float(dv["seq_acc"])
            best_dev_ter = float(dv["ter"])
            best_epoch = epoch
            no_improve_epochs = 0

            ckpt = {
                "model_state": model.state_dict(),
                "vocab": vocab,
                "blank_id": blank_id,
                "config": {
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "name": "bilstm_ctc",
                    "proj_dim": int(model_cfg.get("proj_dim", 192)),
                    "hidden_size": int(model_cfg.get("hidden_size", 192)),
                    "num_layers": int(model_cfg.get("num_layers", 2)),
                    "dropout": float(model_cfg.get("dropout", 0.3)),
                    "normalize": bool(data_cfg.get("normalize", True)),
                    "use_velocity": bool(data_cfg.get("use_velocity", True)),
                },
                "train_config": cfg,
                "best_metric": best_metric,
                "best_metric_value": best_metric_value,
                "best_dev_seq_acc": best_dev_seq_acc,
                "best_dev_ter": best_dev_ter,
                "best_epoch": best_epoch,
            }
            torch.save(ckpt, ckpt_path)
            print(
                f"[train] saved best checkpoint: {ckpt_path} "
                f"(best_{best_metric}={best_metric_value:.4f}, "
                f"seq_acc={best_dev_seq_acc:.4f}, TER={best_dev_ter:.4f}, epoch={best_epoch})"
            )
        else:
            no_improve_epochs += 1

        if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
            print(
                f"[train] early stop at epoch {epoch}: "
                f"no improvement on {best_metric} for {no_improve_epochs} epochs "
                f"(min_delta={early_stop_min_delta})"
            )
            break

    if not ckpt_path.exists():
        raise RuntimeError(f"Best checkpoint not found after training: {ckpt_path}")

    summary_path = ckpt_dir / "summary.json"
    summary = {
        "best_metric": best_metric,
        "best_metric_value": best_metric_value,
        "best_dev_seq_acc": best_dev_seq_acc,
        "best_dev_ter": best_dev_ter,
        "best_epoch": best_epoch,
        "epochs_trained": final_epoch,
        "checkpoint": str(ckpt_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
