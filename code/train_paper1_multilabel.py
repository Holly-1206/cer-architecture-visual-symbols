# -*- coding: utf-8 -*-
"""
train_paper1_multilabel.py
Paper 1 — multi-label architectural visual-lineage recognition.

Run AFTER:
    freeze_paper1_multilabel_dataset.py

Default input:
    <PROJECT_ROOT>/paper1_frozen_dataset

Recommended first runs:
    python train_paper1_multilabel.py --model resnet50
    python train_paper1_multilabel.py --model convnext_tiny

The script:
1) performs 5-fold CV on the expanded-reference domain;
2) tunes one decision threshold per lineage on each validation fold;
3) saves fold checkpoints and metrics;
4) ensembles the 5 folds;
5) evaluates once on the untouched historical and field domains.

Dependencies:
    pip install torch torchvision scikit-learn pillow numpy
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =============================================================================
# CONFIG
# =============================================================================

STYLE_IDS = [f"{i:02d}" for i in range(1, 13)]
STYLE_NAMES = {
    "01": "俄罗斯传统建筑",
    "02": "俄式木构民居",
    "03": "俄罗斯铁路建筑",
    "04": "拜占庭/东正教风格",
    "05": "折衷主义",
    "06": "新艺术运动",
    "07": "巴洛克",
    "08": "中华巴洛克",
    "09": "古典主义/新古典",
    "10": "哥特式",
    "11": "装饰艺术",
    "12": "早期现代主义",
}

DEFAULT_DATA_DIR = Path(
    str(PROJECT_ROOT / "paper1_frozen_dataset_v3")
)
DEFAULT_RESULTS_DIR = Path(str(PROJECT_ROOT / "paper1_model_results"))

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reproducible enough for paper development; some GPU kernels may still vary.
    torch.backends.cudnn.benchmark = True


# =============================================================================
# DATA
# =============================================================================

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class MultiLabelImageDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        p = Path(row["image_path"])

        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            image = self.transform(im)

        target = torch.tensor(
            [float(row[f"y_{sid}"]) for sid in STYLE_IDS],
            dtype=torch.float32,
        )

        return image, target, row["pixel_sha256"]


def make_transforms(image_size: int):
    # Moderate augmentation: preserve architectural geometry while improving
    # robustness to archive/field appearance differences.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.72, 1.0),
            ratio=(0.80, 1.25),
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15)],
            p=0.6,
        ),
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomAffine(
            degrees=3,
            translate=(0.02, 0.02),
            scale=(0.96, 1.04),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.12), antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return train_tf, eval_tf


# =============================================================================
# MODEL
# =============================================================================

def build_model(name: str, num_labels: int = 12) -> nn.Module:
    name = name.lower()

    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model

    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_labels,
        )
        return model

    raise ValueError(f"Unsupported model: {name}")


# =============================================================================
# METRICS
# =============================================================================

def safe_macro_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        vals.append(roc_auc_score(y_true[:, c], probs[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def safe_map(y_true: np.ndarray, probs: np.ndarray) -> float:
    vals = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        vals.append(average_precision_score(y_true[:, c], probs[:, c]))
    return float(np.mean(vals)) if vals else float("nan")


def tune_thresholds(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.15, 0.851, 0.025)

    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)

    for c in range(y_true.shape[1]):
        # If no positive examples in a validation fold, retain 0.5.
        if y_true[:, c].sum() == 0:
            continue

        best_t = 0.5
        best_f1 = -1.0

        for t in grid:
            pred = (probs[:, c] >= t).astype(int)
            f = f1_score(y_true[:, c], pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)

        thresholds[c] = best_t

    return thresholds


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, float]:
    pred = (probs >= thresholds[None, :]).astype(int)

    return {
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, pred, average="micro", zero_division=0),
        "macro_precision": precision_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "samples_jaccard": jaccard_score(
            y_true, pred, average="samples", zero_division=0
        ),
        "hamming_loss": hamming_loss(y_true, pred),
        "mAP": safe_map(y_true, probs),
        "macro_AUROC": safe_macro_auroc(y_true, probs),
    }


def per_label_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> List[Dict[str, float]]:
    rows = []
    pred = (probs >= thresholds[None, :]).astype(int)

    for i, sid in enumerate(STYLE_IDS):
        yt = y_true[:, i]
        yp = pred[:, i]

        ap = (
            average_precision_score(yt, probs[:, i])
            if yt.sum() > 0 else float("nan")
        )
        auc = (
            roc_auc_score(yt, probs[:, i])
            if len(np.unique(yt)) >= 2 else float("nan")
        )

        rows.append({
            "style_id": sid,
            "style_name": STYLE_NAMES[sid],
            "n_positive": int(yt.sum()),
            "threshold": float(thresholds[i]),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
            "AP": ap,
            "AUROC": auc,
        })
    return rows


# =============================================================================
# TRAIN / PREDICT
# =============================================================================

def make_pos_weight(rows: List[Dict[str, str]]) -> torch.Tensor:
    y = np.array(
        [[int(r[f"y_{sid}"]) for sid in STYLE_IDS] for r in rows],
        dtype=np.float32,
    )
    pos = y.sum(axis=0)
    neg = len(y) - pos
    weight = neg / np.maximum(pos, 1.0)

    # Avoid pathological weights while retaining imbalance correction.
    weight = np.clip(weight, 1.0, 12.0)
    return torch.tensor(weight, dtype=torch.float32)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()

    ys = []
    ps = []
    ids = []

    for images, targets, pixel_ids in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.sigmoid(logits)

        ys.append(targets.numpy())
        ps.append(probs.cpu().numpy())
        ids.extend(list(pixel_ids))

    return (
        np.concatenate(ys, axis=0),
        np.concatenate(ps, axis=0),
        ids,
    )


def train_one_fold(
    model_name: str,
    train_rows: List[Dict[str, str]],
    val_rows: List[Dict[str, str]],
    fold: int,
    args,
    run_dir: Path,
    device: torch.device,
):
    train_tf, eval_tf = make_transforms(args.image_size)

    train_ds = MultiLabelImageDataset(train_rows, train_tf)
    val_ds = MultiLabelImageDataset(val_rows, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    model = build_model(model_name).to(device)

    pos_weight = make_pos_weight(train_rows).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.02,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_macro_f1 = -1.0
    best_epoch = -1
    patience_used = 0
    ckpt = run_dir / f"fold_{fold}_best.pt"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs

        scheduler.step()

        yv, pv, _ = predict(model, val_loader, device)
        thresholds = tune_thresholds(yv, pv)
        metrics = compute_metrics(yv, pv, thresholds)

        train_loss = running_loss / max(n_seen, 1)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(rec)

        print(
            f"[fold {fold}] epoch {epoch:03d} "
            f"loss={train_loss:.4f} "
            f"macroF1={metrics['macro_f1']:.4f} "
            f"microF1={metrics['micro_f1']:.4f} "
            f"mAP={metrics['mAP']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1 + args.min_delta:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch
            patience_used = 0

            torch.save(
                {
                    "model_name": model_name,
                    "fold": fold,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "thresholds": thresholds.tolist(),
                    "style_ids": STYLE_IDS,
                    "style_names": STYLE_NAMES,
                    "image_size": args.image_size,
                    "val_metrics": metrics,
                },
                ckpt,
            )
        else:
            patience_used += 1

        if patience_used >= args.patience:
            print(f"[fold {fold}] early stop at epoch {epoch}")
            break

    # Save history
    write_rows_csv(
        run_dir / f"fold_{fold}_history.csv",
        history,
    )

    # Reload best checkpoint and obtain final validation predictions.
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])

    yv, pv, ids = predict(model, val_loader, device)
    thresholds = np.array(state["thresholds"], dtype=np.float32)
    final_metrics = compute_metrics(yv, pv, thresholds)
    per_label = per_label_metrics(yv, pv, thresholds)

    write_rows_csv(
        run_dir / f"fold_{fold}_per_label_metrics.csv",
        per_label,
    )

    pred_rows = []
    for i, pid in enumerate(ids):
        rr = {"pixel_sha256": pid}
        for j, sid in enumerate(STYLE_IDS):
            rr[f"y_{sid}"] = int(yv[i, j])
            rr[f"p_{sid}"] = float(pv[i, j])
        pred_rows.append(rr)

    write_rows_csv(
        run_dir / f"fold_{fold}_validation_predictions.csv",
        pred_rows,
    )

    return {
        "fold": fold,
        "best_epoch": best_epoch,
        **final_metrics,
    }, ckpt, thresholds


def write_rows_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# =============================================================================
# EXTERNAL ENSEMBLE
# =============================================================================

@torch.no_grad()
def ensemble_external(
    ckpts: List[Path],
    thresholds_by_fold: List[np.ndarray],
    rows: List[Dict[str, str]],
    model_name: str,
    args,
    device: torch.device,
    out_dir: Path,
    domain_name: str,
):
    _, eval_tf = make_transforms(args.image_size)
    ds = MultiLabelImageDataset(rows, eval_tf)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    prob_stack = []
    y_true = None
    ids = None

    for ckpt in ckpts:
        state = torch.load(ckpt, map_location=device)
        model = build_model(model_name).to(device)
        model.load_state_dict(state["state_dict"])

        y, p, this_ids = predict(model, loader, device)

        if y_true is None:
            y_true = y
            ids = this_ids

        prob_stack.append(p)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = np.mean(np.stack(prob_stack, axis=0), axis=0)

    # Frozen threshold rule: median of validation-tuned fold thresholds.
    thresholds = np.median(np.stack(thresholds_by_fold, axis=0), axis=0)

    metrics = compute_metrics(y_true, probs, thresholds)
    per_label = per_label_metrics(y_true, probs, thresholds)

    write_rows_csv(
        out_dir / f"{domain_name}_per_label_metrics.csv",
        per_label,
    )

    pred_rows = []
    for i, pid in enumerate(ids):
        rr = {"pixel_sha256": pid}
        for j, sid in enumerate(STYLE_IDS):
            rr[f"y_{sid}"] = int(y_true[i, j])
            rr[f"p_{sid}"] = float(probs[i, j])
            rr[f"pred_{sid}"] = int(probs[i, j] >= thresholds[j])
        pred_rows.append(rr)

    write_rows_csv(
        out_dir / f"{domain_name}_ensemble_predictions.csv",
        pred_rows,
    )

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument(
        "--model",
        choices=["resnet50", "convnext_tiny"],
        default="convnext_tiny",
    )
    p.add_argument("--image-size", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260916)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ref_path = args.data_dir / "02_reference_development_5fold.csv"
    hist_path = args.data_dir / "03_historical_external.csv"
    field_path = args.data_dir / "04_field_external.csv"

    reference = read_csv(ref_path)
    historical = read_csv(hist_path)
    field = read_csv(field_path)

    run_dir = args.results_dir / args.model
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PAPER 1 MULTI-LABEL LINEAGE RECOGNITION")
    print("=" * 100)
    print(f"Model:       {args.model}")
    print(f"Reference:   {len(reference):,}")
    print(f"Historical:  {len(historical):,}")
    print(f"Field:       {len(field):,}")
    print("=" * 100)

    fold_metrics = []
    ckpts = []
    thresholds_all = []

    for fold in range(5):
        train_rows = [r for r in reference if int(r["fold"]) != fold]
        val_rows = [r for r in reference if int(r["fold"]) == fold]

        print()
        print(f"FOLD {fold}: train={len(train_rows):,}, val={len(val_rows):,}")

        m, ckpt, thr = train_one_fold(
            args.model,
            train_rows,
            val_rows,
            fold,
            args,
            run_dir,
            device,
        )

        fold_metrics.append(m)
        ckpts.append(ckpt)
        thresholds_all.append(thr)

    write_rows_csv(run_dir / "cv_fold_metrics.csv", fold_metrics)

    # CV mean/std
    cv_summary = {}
    metric_names = [
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "samples_jaccard",
        "hamming_loss",
        "mAP",
        "macro_AUROC",
    ]
    for k in metric_names:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        cv_summary[f"{k}_mean"] = float(np.nanmean(vals))
        cv_summary[f"{k}_sd"] = float(np.nanstd(vals, ddof=1))

    # External evaluation only AFTER all development folds are completed.
    hist_metrics = ensemble_external(
        ckpts,
        thresholds_all,
        historical,
        args.model,
        args,
        device,
        run_dir,
        "historical_external",
    )

    field_metrics = ensemble_external(
        ckpts,
        thresholds_all,
        field,
        args.model,
        args,
        device,
        run_dir,
        "field_external",
    )

    final = {
        "model": args.model,
        "reference_n": len(reference),
        "historical_n": len(historical),
        "field_n": len(field),
        **{f"cv_{k}": v for k, v in cv_summary.items()},
        **{f"historical_{k}": v for k, v in hist_metrics.items()},
        **{f"field_{k}": v for k, v in field_metrics.items()},
    }

    (run_dir / "final_metrics.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FINAL")
    print("=" * 100)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    print(f"Saved to: {run_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
