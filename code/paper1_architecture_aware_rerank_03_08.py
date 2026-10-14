# -*- coding: utf-8 -*-
r"""
paper1_architecture_aware_rerank_03_08.py
=========================================

Purpose
-------
Architecture-aware re-ranking for two problematic lineages:

03  Russian Railway
08  Chinese Baroque

Why needed
----------
Pure DINO prototype similarity sometimes picks visually similar but
architecturally unsuitable images, e.g. locomotives instead of buildings
for lineage 03, or overly generic "Chinese-style" imagery for lineage 08.

This script re-ranks candidates by combining:
1) DINO prototype similarity
2) A–D architectural-symbol intensity
3) lineage-specific morphology logic
4) penalties for non-building / over-contextual imagery

Outputs
-------
<PROJECT_ROOT>/paper1_photo_candidates_03_08_rerank/
    00_summary.txt
    01_03_08_architecture_aware_candidates.csv
    contact_sheets\
        03_reference_architecture_aware.jpg
        03_historical_architecture_aware.jpg
        03_field_architecture_aware.jpg
        08_reference_architecture_aware.jpg
        08_historical_architecture_aware.jpg
        08_field_architecture_aware.jpg

Dependencies
------------
pip install pandas numpy pillow

Run
---
python paper1_architecture_aware_rerank_03_08.py
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import math
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(str(PROJECT_ROOT))
DINO_DIR = ROOT / "paper1_dinov2_results"
FROZEN_DIR = ROOT / "paper1_frozen_dataset_v3"

OUT_DIR = ROOT / "paper1_photo_candidates_03_08_rerank"
CONTACT_DIR = OUT_DIR / "contact_sheets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20

STYLE_NAMES = {
    "01": "Russian Traditional",
    "02": "Russian Timber Dwelling",
    "03": "Russian Railway",
    "04": "Byzantine / Orthodox",
    "05": "Eclecticism",
    "06": "Art Nouveau",
    "07": "Baroque",
    "08": "Chinese Baroque",
    "09": "Classical / Neoclassical",
    "10": "Gothic",
    "11": "Art Deco",
    "12": "Early Modernism",
}

TARGET_STYLES = ["03", "08"]
TARGET_DOMAINS = ["reference", "historical", "field"]

# contact-sheet palette
BG = "#FBFAF7"
TEXT = "#2B2B2B"
REF = "#68747E"
HIST = "#B5654D"
FIELD = "#2D6F6A"

# =============================================================================
# SYMBOL DEFINITIONS
# =============================================================================

AD_PREFIXES = ("A", "B", "C", "D")
E_PREFIXES = ("E",)

# For 03 Russian Railway:
# architectural railway-related built elements
RAILWAY_BUILT_E = {
    "E1", "E2", "E4", "E5", "E6", "E7", "E8", "E10", "E11", "E12", "E13"
}
# stronger "context / object" indicators that often drift away from buildingness
RAILWAY_OBJECTISH_E = {
    "E3", "E14", "E15"
}
# architectural morphology that often helps identify railway architecture
RAILWAY_ARCH_KEYS = {
    "B14", "B15", "B16",   # entrance / steps / fanlight
    "C13", "C14", "C15", "C18", "C20",  # structural/material cues
    "D1", "D21"
}

# For 08 Chinese Baroque:
CHINESE_KEYS = {
    "A12", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D18"
}
WESTERN_KEYS = {
    "A9", "B1", "B2", "B10", "B11",
    "C9", "C10", "C20",
    "D1", "D3", "D4", "D5"
}

# =============================================================================
# IO
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require(path), encoding="utf-8-sig")


def load_domain(meta_name: str, emb_name: str, domain: str):
    meta = read_csv(DINO_DIR / meta_name)
    emb = np.load(require(DINO_DIR / emb_name))

    if len(meta) != len(emb):
        raise RuntimeError(
            f"Row mismatch for {meta_name}: meta={len(meta)}, emb={len(emb)}"
        )

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)

    meta = meta.copy()
    meta["domain"] = domain
    return meta, emb


def load_master() -> pd.DataFrame:
    m = read_csv(FROZEN_DIR / "01_unique_image_multilabel_master.csv").copy()

    role_to_domain = {
        "development_reference": "reference",
        "external_historical": "historical",
        "external_field": "field",
    }
    m = m[m["dataset_role"].isin(role_to_domain)].copy()
    m["domain"] = m["dataset_role"].map(role_to_domain)
    return m


# =============================================================================
# DATA HELPERS
# =============================================================================

def style_id_clean(x) -> str:
    try:
        return f"{int(float(x)):02d}"
    except Exception:
        return str(x).strip().zfill(2)


def y_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df[f"y_{i:02d}"].astype(float).astype(int).to_numpy()
        for i in range(1, 13)
    ])


def parse_symbol_counts(value) -> Dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        obj = json.loads(value)
        return {str(k): float(v) for k, v in obj.items() if float(v) > 0}
    except Exception:
        return {}


def count_set(d: Dict[str, float], keys: set[str]) -> float:
    return sum(v for k, v in d.items() if k in keys)


def count_prefix(d: Dict[str, float], prefixes: Tuple[str, ...]) -> float:
    return sum(v for k, v in d.items() if k.startswith(prefixes))


def get_merge_key(df: pd.DataFrame) -> str:
    if "pixel_sha256" in df.columns:
        return "pixel_sha256"
    return "image_path"


# =============================================================================
# PROTOTYPES
# =============================================================================

def build_reference_prototypes(ref_meta: pd.DataFrame, ref_emb: np.ndarray) -> Dict[str, np.ndarray]:
    Y = y_matrix(ref_meta)
    single = Y.sum(axis=1) == 1

    proto = {}

    for j in range(12):
        sid = f"{j+1:02d}"
        mask = single & (Y[:, j] == 1)

        if mask.sum() < 5:
            mask = (Y[:, j] == 1)

        p = ref_emb[mask].mean(axis=0)
        p = p / max(np.linalg.norm(p), 1e-12)
        proto[sid] = p

    return proto


# =============================================================================
# SCORING
# =============================================================================

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def feature_bundle(symbol_counts: Dict[str, float]) -> Dict[str, float]:
    total = sum(symbol_counts.values())

    ad_total = count_prefix(symbol_counts, AD_PREFIXES)
    e_total = count_prefix(symbol_counts, E_PREFIXES)

    a_total = count_prefix(symbol_counts, ("A",))
    b_total = count_prefix(symbol_counts, ("B",))
    c_total = count_prefix(symbol_counts, ("C",))
    d_total = count_prefix(symbol_counts, ("D",))

    buildingness = safe_div((0.7 * a_total + 1.0 * b_total + 1.0 * c_total + 1.0 * d_total), (total + 1e-9))
    ad_ratio = safe_div(ad_total, (total + 1e-9))
    e_ratio = safe_div(e_total, (total + 1e-9))

    railway_built = safe_div(count_set(symbol_counts, RAILWAY_BUILT_E), (total + 1e-9))
    railway_objectish = safe_div(count_set(symbol_counts, RAILWAY_OBJECTISH_E), (total + 1e-9))
    railway_arch = safe_div(count_set(symbol_counts, RAILWAY_ARCH_KEYS), (total + 1e-9))

    chinese = safe_div(count_set(symbol_counts, CHINESE_KEYS), (total + 1e-9))
    western = safe_div(count_set(symbol_counts, WESTERN_KEYS), (total + 1e-9))
    hybrid_balance = 2.0 * min(chinese, western)
    hybrid_gap = abs(chinese - western)

    pure_context_penalty = 1.0 if (ad_total < 2 and e_total >= 3) else 0.0
    weak_building_penalty = 1.0 if buildingness < 0.25 else 0.0

    return {
        "total_symbol_count": total,
        "ad_total": ad_total,
        "e_total": e_total,
        "buildingness": clamp01(buildingness),
        "ad_ratio": clamp01(ad_ratio),
        "e_ratio": clamp01(e_ratio),
        "railway_built": clamp01(railway_built),
        "railway_objectish": clamp01(railway_objectish),
        "railway_arch": clamp01(railway_arch),
        "chinese": clamp01(chinese),
        "western": clamp01(western),
        "hybrid_balance": clamp01(hybrid_balance),
        "hybrid_gap": clamp01(hybrid_gap),
        "pure_context_penalty": pure_context_penalty,
        "weak_building_penalty": weak_building_penalty,
    }


def score_architecture_aware(
    sid: str,
    proto_similarity: float,
    feats: Dict[str, float],
) -> Dict[str, float]:
    """
    Output:
    - final_score
    - decomposed components
    """

    s = clamp01((proto_similarity + 1.0) / 2.0) if proto_similarity < 0 else clamp01(proto_similarity)

    if sid == "03":
        # Russian Railway Architecture
        # Goal: reward railway-related architecture, not pure rolling-stock / pure context.
        final = (
            0.50 * s
            + 0.18 * feats["buildingness"]
            + 0.16 * feats["ad_ratio"]
            + 0.12 * feats["railway_built"]
            + 0.10 * feats["railway_arch"]
            - 0.18 * feats["railway_objectish"]
            - 0.12 * feats["pure_context_penalty"]
            - 0.08 * feats["weak_building_penalty"]
        )

    elif sid == "08":
        # Chinese Baroque
        # Goal: reward hybridity (Chinese + Western together), not generic pure Chinese.
        final = (
            0.52 * s
            + 0.18 * feats["buildingness"]
            + 0.12 * feats["ad_ratio"]
            + 0.22 * feats["hybrid_balance"]
            - 0.10 * feats["hybrid_gap"]
            - 0.08 * feats["weak_building_penalty"]
        )
    else:
        raise ValueError(f"Unsupported sid: {sid}")

    out = dict(feats)
    out["prototype_similarity"] = float(proto_similarity)
    out["final_score"] = float(final)
    return out


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im.copy()


def crop_fill(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = im.size
    target_ratio = target_w / target_h
    ratio = w / h

    if ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))

    return im.resize((target_w, target_h), Image.Resampling.LANCZOS)


def basename_short(path: str, max_chars: int = 46) -> str:
    name = Path(path).name
    if len(name) <= max_chars:
        return name
    return name[: max_chars - 1] + "…"


def make_contact_sheet(
    records: pd.DataFrame,
    out_path: Path,
    title: str,
    domain: str,
    ncols: int = 4,
    cell_w: int = 360,
    cell_h: int = 250,
):
    if records.empty:
        return

    n = len(records)
    nrows = math.ceil(n / ncols)

    top_h = 72
    text_h = 110
    pad = 14

    canvas_w = pad + ncols * (cell_w + pad)
    canvas_h = top_h + nrows * (cell_h + text_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

    draw.text((pad, 18), title, fill=TEXT, font=font_title)

    border_color = {"reference": REF, "historical": HIST, "field": FIELD}[domain]

    for idx, (_, r) in enumerate(records.iterrows()):
        rr = idx // ncols
        cc = idx % ncols

        x0 = pad + cc * (cell_w + pad)
        y0 = top_h + rr * (cell_h + text_h + pad)

        try:
            im = crop_fill(open_rgb(str(r["image_path"])), cell_w, cell_h)
            canvas.paste(im, (x0, y0))
        except Exception as e:
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline="#999999", width=2)
            draw.text((x0 + 8, y0 + 8), f"IMAGE ERROR\n{e}", fill="#8A2D2D", font=font_text)

        draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], outline=border_color, width=5)

        lines = [
            f"Rank {int(r['rank'])} | final={float(r['final_score']):.3f}",
            f"proto={float(r['prototype_similarity']):.3f} | build={float(r['buildingness']):.3f}",
        ]

        if r["style_id"] == "03":
            lines.append(
                f"rail-built={float(r['railway_built']):.3f} | "
                f"rail-arch={float(r['railway_arch']):.3f} | "
                f"obj-pen={float(r['railway_objectish']):.3f}"
            )
        elif r["style_id"] == "08":
            lines.append(
                f"hybrid={float(r['hybrid_balance']):.3f} | "
                f"CN={float(r['chinese']):.3f} | "
                f"W={float(r['western']):.3f}"
            )

        lines.append(basename_short(str(r["image_path"])))

        draw.multiline_text(
            (x0 + 4, y0 + cell_h + 5),
            "\n".join(lines),
            fill=TEXT,
            font=font_text,
            spacing=3,
        )

    canvas.save(out_path, quality=95)


# =============================================================================
# MAIN LOGIC
# =============================================================================

def main():
    print("=" * 100)
    print("ARCHITECTURE-AWARE RE-RANKING FOR 03 / 08")
    print("=" * 100)

    master = load_master()

    ref_meta, ref_emb = load_domain(
        "01_reference_embedding_meta.csv",
        "embeddings_reference.npy",
        "reference",
    )
    hist_meta, hist_emb = load_domain(
        "02_historical_embedding_meta.csv",
        "embeddings_historical.npy",
        "historical",
    )
    field_meta, field_emb = load_domain(
        "03_field_embedding_meta.csv",
        "embeddings_field.npy",
        "field",
    )

    prototypes = build_reference_prototypes(ref_meta, ref_emb)

    dino_map = {
        "reference": (ref_meta, ref_emb),
        "historical": (hist_meta, hist_emb),
        "field": (field_meta, field_emb),
    }

    # Merge symbol-count data from master into each meta table
    master_key = get_merge_key(master)

    if master_key not in master.columns:
        raise RuntimeError(f"Master merge key not found: {master_key}")

    master_merge_cols = [master_key, "image_path", "symbol_counts_json", "domain"]
    master_small = master[master_merge_cols].drop_duplicates(subset=[master_key]).copy()

    results = []

    for domain in TARGET_DOMAINS:
        meta, emb = dino_map[domain]
        meta = meta.copy()

        meta_key = master_key if master_key in meta.columns else "image_path"
        if meta_key not in meta.columns:
            raise RuntimeError(
                f"Neither '{master_key}' nor 'image_path' exists in {domain} meta."
            )

        meta = meta.merge(
            master_small[[meta_key, "symbol_counts_json"]],
            on=meta_key,
            how="left",
        )

        Y = y_matrix(meta)

        for sid in TARGET_STYLES:
            j = int(sid) - 1
            mask = (Y[:, j] == 1)

            sub = meta.loc[mask].copy()
            sub_emb = emb[mask]

            if len(sub) == 0:
                continue

            proto = prototypes[sid]
            sims = sub_emb @ proto

            rows = []

            for row_idx, (_, r) in enumerate(sub.iterrows()):
                sym = parse_symbol_counts(r.get("symbol_counts_json", ""))

                feats = feature_bundle(sym)
                scored = score_architecture_aware(
                    sid=sid,
                    proto_similarity=float(sims[row_idx]),
                    feats=feats,
                )

                rows.append({
                    "style_id": sid,
                    "style_name": STYLE_NAMES[sid],
                    "domain": domain,
                    "image_path": r["image_path"],
                    meta_key: r[meta_key],
                    **scored,
                })

            cand = pd.DataFrame(rows).sort_values(
                ["final_score", "prototype_similarity"],
                ascending=[False, False],
            ).reset_index(drop=True)

            cand["rank"] = np.arange(1, len(cand) + 1)

            top = cand.head(TOP_N).copy()
            results.append(top)

            make_contact_sheet(
                top,
                CONTACT_DIR / f"{sid}_{domain}_architecture_aware.jpg",
                title=f"{sid} {STYLE_NAMES[sid]} | {domain} | architecture-aware rerank",
                domain=domain,
                ncols=4,
            )

    if not results:
        raise RuntimeError("No candidates generated.")

    out = pd.concat(results, ignore_index=True)
    out.to_csv(
        OUT_DIR / "01_03_08_architecture_aware_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # small summary
    summary_lines = [
        "ARCHITECTURE-AWARE RE-RANKING COMPLETE",
        "",
        f"Output directory: {OUT_DIR}",
        "",
        "Generated contact sheets:",
    ]
    for sid in TARGET_STYLES:
        for domain in TARGET_DOMAINS:
            summary_lines.append(
                f"  - {CONTACT_DIR / f'{sid}_{domain}_architecture_aware.jpg'}"
            )

    summary_lines += [
        "",
        "Interpretation tips:",
        "03: prefer candidates with railway-building morphology, not pure locomotives.",
        "08: prefer candidates that visibly combine Chinese and Western/Baroque cues.",
        "",
        "Next step:",
        "Review the contact sheets and pick 1 best image for each relevant role.",
    ]

    (OUT_DIR / "00_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()