# -*- coding: utf-8 -*-
r"""
paper1_dinov2_crossdomain.py

Paper 1 — DINOv2 cross-domain visual-lineage analysis
-----------------------------------------------------
This script is the recommended NEXT STEP after ConvNeXt revealed a strong
reference-to-archive/field domain shift.

It does NOT fine-tune on historical or field data.
It uses DINOv2 as a frozen visual encoder, then evaluates whether a generic
foundation-model representation transfers better across domains.

Inputs
------
<PROJECT_ROOT>/paper1_frozen_dataset_v3/
    02_reference_development_5fold.csv
    03_historical_external.csv
    04_field_external.csv

Outputs
-------
<PROJECT_ROOT>/paper1_dinov2_results/
    00_summary.json
    01_reference_embedding_meta.csv
    02_historical_embedding_meta.csv
    03_field_embedding_meta.csv
    embeddings_reference.npy
    embeddings_historical.npy
    embeddings_field.npy
    04_prototype_similarity_reference.csv
    05_domain_centroid_alignment.csv
    06_prototype_external_metrics.csv
    07_linear_probe_external_metrics.csv
    08_per_lineage_external_metrics.csv
    09_reference_pca2d.csv
    10_historical_pca2d.csv
    11_field_pca2d.csv

Dependencies
------------
pip install torch torchvision scikit-learn pandas pillow numpy

First run requires Internet access because torch.hub downloads the official
facebookresearch/dinov2 model and weights. They are cached locally afterwards.

Recommended command
-------------------
python paper1_dinov2_crossdomain.py --model dinov2_vitb14 --image-size 518 --batch-size 16

If GPU memory is limited:
python paper1_dinov2_crossdomain.py --model dinov2_vits14 --image-size 448 --batch-size 16
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    label_ranking_average_precision_score,
    label_ranking_loss,
    coverage_error,
)

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

DEFAULT_DATA = Path(str(PROJECT_ROOT / "paper1_frozen_dataset_v3"))
DEFAULT_OUT = Path(str(PROJECT_ROOT / "paper1_dinov2_results"))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class ImgDataset(Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        with Image.open(r["image_path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            x = self.tf(im)
        y = torch.tensor(
            [float(r[f"y_{sid}"]) for sid in STYLE_IDS],
            dtype=torch.float32
        )
        return x, y, r["pixel_sha256"]


def make_tf(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_dinov2(name: str, device):
    print(f"Loading {name} from official facebookresearch/dinov2 torch.hub ...")
    model = torch.hub.load(
        "facebookresearch/dinov2",
        name,
        pretrained=True,
        trust_repo=True,
        skip_validation=True
    )
    model = model.to(device).eval()
    return model


@torch.no_grad()
def extract(model, rows, tf, batch_size, workers, device):
    ds = ImgDataset(rows, tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )

    feats, ys, ids = [], [], []

    for b, (x, y, pid) in enumerate(dl, 1):
        x = x.to(device, non_blocking=True)

        with torch.amp.autocast(
            "cuda", enabled=(device.type == "cuda")
        ):
            z = model(x)

        # Official DINOv2 forward returns the CLS embedding tensor.
        if isinstance(z, dict):
            if "x_norm_clstoken" in z:
                z = z["x_norm_clstoken"]
            else:
                # fallback: first tensor value
                z = next(v for v in z.values() if torch.is_tensor(v))

        feats.append(z.float().cpu().numpy())
        ys.append(y.numpy())
        ids.extend(list(pid))

        if b % 20 == 0:
            print(f"  batches processed: {b}/{len(dl)}")

    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(ys, axis=0)

    # L2 normalization is useful for cosine/prototype analysis.
    X = normalize(X, norm="l2")

    return X, Y, ids


def safe_eval(y, scores):
    aucs, aps = [], []
    per = []

    for j, sid in enumerate(STYLE_IDS):
        yt = y[:, j]
        sc = scores[:, j]

        auc = np.nan
        ap = np.nan

        if len(np.unique(yt)) == 2:
            auc = roc_auc_score(yt, sc)
            aucs.append(auc)

        if yt.sum() > 0:
            ap = average_precision_score(yt, sc)
            aps.append(ap)

        per.append({
            "style_id": sid,
            "style_name": STYLE_NAMES[sid],
            "n_positive": int(yt.sum()),
            "prevalence": float(yt.mean()),
            "AP": ap,
            "AUROC": auc,
        })

    return {
        "macro_AUROC": float(np.nanmean(aucs)),
        "mAP": float(np.nanmean(aps)),
        "LRAP": float(label_ranking_average_precision_score(y, scores)),
        "ranking_loss": float(label_ranking_loss(y, scores)),
        "coverage_error": float(coverage_error(y, scores)),
    }, per


def prototypes_from_reference(X, Y):
    """
    Use only single-lineage reference images when possible.
    This preserves the idea of a canonical visual prototype.
    """
    single = Y.sum(axis=1) == 1
    P = []

    for j in range(Y.shape[1]):
        mask = single & (Y[:, j] == 1)

        if mask.sum() < 5:
            mask = Y[:, j] == 1

        p = X[mask].mean(axis=0)
        p = p / max(np.linalg.norm(p), 1e-12)
        P.append(p)

    return np.stack(P, axis=0)


def cosine_scores(X, P):
    # both X and P are L2-normalized
    return X @ P.T


def pairwise_proto_similarity(P):
    return P @ P.T


def domain_centroid_alignment(Xr, Yr, Xt, Yt, target_domain):
    rows = []

    for j, sid in enumerate(STYLE_IDS):
        mr = Yr[:, j] == 1
        mt = Yt[:, j] == 1

        if mr.sum() == 0 or mt.sum() == 0:
            sim = np.nan
        else:
            cr = Xr[mr].mean(axis=0)
            ct = Xt[mt].mean(axis=0)
            cr /= max(np.linalg.norm(cr), 1e-12)
            ct /= max(np.linalg.norm(ct), 1e-12)
            sim = float(np.dot(cr, ct))

        rows.append({
            "style_id": sid,
            "style_name": STYLE_NAMES[sid],
            "target_domain": target_domain,
            "n_reference_positive": int(mr.sum()),
            "n_target_positive": int(mt.sum()),
            "centroid_cosine_similarity": sim,
        })

    return rows


def linear_probe(Xr, Yr, Xt):
    """
    Transparent frozen-feature probe:
    - DINOv2 is NOT fine-tuned.
    - only a one-vs-rest logistic classifier is learned on reference embeddings.
    """
    clf = OneVsRestClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=20260917,
            ),
        ),
        n_jobs=-1,
    )
    clf.fit(Xr, Yr)
    return clf.predict_proba(Xt), clf


def save_meta(path, rows, ids):
    out = []
    by_id = {r["pixel_sha256"]: r for r in rows}

    for i, pid in enumerate(ids):
        r = by_id[pid]
        x = {
            "embedding_row": i,
            "pixel_sha256": pid,
            "source": r["source"],
            "style_ids": r["style_ids"],
            "style_names": r["style_names"],
            "multi_lineage": r["multi_lineage"],
            "image_path": r["image_path"],
        }
        for sid in STYLE_IDS:
            x[f"y_{sid}"] = r[f"y_{sid}"]
        out.append(x)

    pd.DataFrame(out).to_csv(path, index=False, encoding="utf-8-sig")


def pca_coordinates(X_all, n_ref, n_hist, ref_ids, hist_ids, field_ids, outdir):
    pca = PCA(n_components=2, random_state=20260917)
    coords = pca.fit_transform(X_all)

    a = 0
    cr = coords[a:a+n_ref]; a += n_ref
    ch = coords[a:a+n_hist]; a += n_hist
    cf = coords[a:]

    pd.DataFrame({
        "pixel_sha256": ref_ids,
        "PC1": cr[:, 0],
        "PC2": cr[:, 1],
    }).to_csv(outdir / "09_reference_pca2d.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "pixel_sha256": hist_ids,
        "PC1": ch[:, 0],
        "PC2": ch[:, 1],
    }).to_csv(outdir / "10_historical_pca2d.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({
        "pixel_sha256": field_ids,
        "PC1": cf[:, 0],
        "PC2": cf[:, 1],
    }).to_csv(outdir / "11_field_pca2d.csv", index=False, encoding="utf-8-sig")

    return pca.explained_variance_ratio_.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--model",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
        default="dinov2_vitb14",
    )
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reference = read_csv(args.data_dir / "02_reference_development_5fold.csv")
    historical = read_csv(args.data_dir / "03_historical_external.csv")
    field = read_csv(args.data_dir / "04_field_external.csv")

    print("=" * 100)
    print("DINOv2 CROSS-DOMAIN VISUAL GENEALOGY")
    print("=" * 100)
    print(f"Reference:  {len(reference)}")
    print(f"Historical: {len(historical)}")
    print(f"Field:      {len(field)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    model = load_dinov2(args.model, device)
    tf = make_tf(args.image_size)

    print("\n[1/3] Reference embeddings")
    Xr, Yr, idr = extract(
        model, reference, tf, args.batch_size, args.workers, device
    )

    print("\n[2/3] Historical embeddings")
    Xh, Yh, idh = extract(
        model, historical, tf, args.batch_size, args.workers, device
    )

    print("\n[3/3] Field embeddings")
    Xf, Yf, idf = extract(
        model, field, tf, args.batch_size, args.workers, device
    )

    np.save(args.out_dir / "embeddings_reference.npy", Xr)
    np.save(args.out_dir / "embeddings_historical.npy", Xh)
    np.save(args.out_dir / "embeddings_field.npy", Xf)

    save_meta(args.out_dir / "01_reference_embedding_meta.csv", reference, idr)
    save_meta(args.out_dir / "02_historical_embedding_meta.csv", historical, idh)
    save_meta(args.out_dir / "03_field_embedding_meta.csv", field, idf)

    # -------------------------------------------------------------------------
    # A. Prototype analysis
    # -------------------------------------------------------------------------
    P = prototypes_from_reference(Xr, Yr)

    proto_sim = pairwise_proto_similarity(P)
    sim_df = pd.DataFrame(
        proto_sim,
        index=[f"{s}_{STYLE_NAMES[s]}" for s in STYLE_IDS],
        columns=[f"{s}_{STYLE_NAMES[s]}" for s in STYLE_IDS],
    )
    sim_df.to_csv(
        args.out_dir / "04_prototype_similarity_reference.csv",
        encoding="utf-8-sig"
    )

    Sh = cosine_scores(Xh, P)
    Sf = cosine_scores(Xf, P)

    hist_proto_metrics, hist_proto_per = safe_eval(Yh, Sh)
    field_proto_metrics, field_proto_per = safe_eval(Yf, Sf)

    proto_rows = [
        {"method": "DINOv2_reference_prototype", "domain": "historical", **hist_proto_metrics},
        {"method": "DINOv2_reference_prototype", "domain": "field", **field_proto_metrics},
    ]
    pd.DataFrame(proto_rows).to_csv(
        args.out_dir / "06_prototype_external_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # B. Frozen linear probe
    # -------------------------------------------------------------------------
    print("\nFitting frozen-feature logistic linear probe on reference only...")
    Ph, clf = linear_probe(Xr, Yr, Xh)
    Pf = clf.predict_proba(Xf)

    hist_probe_metrics, hist_probe_per = safe_eval(Yh, Ph)
    field_probe_metrics, field_probe_per = safe_eval(Yf, Pf)

    pd.DataFrame([
        {"method": "DINOv2_frozen_linear_probe", "domain": "historical", **hist_probe_metrics},
        {"method": "DINOv2_frozen_linear_probe", "domain": "field", **field_probe_metrics},
    ]).to_csv(
        args.out_dir / "07_linear_probe_external_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # C. Per-lineage metrics
    # -------------------------------------------------------------------------
    per_rows = []

    for domain, method, rows in [
        ("historical", "prototype", hist_proto_per),
        ("field", "prototype", field_proto_per),
        ("historical", "linear_probe", hist_probe_per),
        ("field", "linear_probe", field_probe_per),
    ]:
        for r in rows:
            per_rows.append({
                "domain": domain,
                "method": method,
                **r,
            })

    pd.DataFrame(per_rows).to_csv(
        args.out_dir / "08_per_lineage_external_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # D. Domain centroid alignment
    # -------------------------------------------------------------------------
    align = []
    align += domain_centroid_alignment(Xr, Yr, Xh, Yh, "historical")
    align += domain_centroid_alignment(Xr, Yr, Xf, Yf, "field")

    pd.DataFrame(align).to_csv(
        args.out_dir / "05_domain_centroid_alignment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # E. Shared PCA only for visualization QA; later UMAP can be added
    # -------------------------------------------------------------------------
    allX = np.vstack([Xr, Xh, Xf])
    evr = pca_coordinates(
        allX,
        len(Xr),
        len(Xh),
        idr,
        idh,
        idf,
        args.out_dir,
    )

    summary = {
        "model": args.model,
        "image_size": args.image_size,
        "reference_n": len(reference),
        "historical_n": len(historical),
        "field_n": len(field),
        "embedding_dim": int(Xr.shape[1]),
        "prototype_historical": hist_proto_metrics,
        "prototype_field": field_proto_metrics,
        "linear_probe_historical": hist_probe_metrics,
        "linear_probe_field": field_probe_metrics,
        "pca_explained_variance_ratio": evr,
    }

    (args.out_dir / "00_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("COMPLETE")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput: {args.out_dir}")


if __name__ == "__main__":
    main()
