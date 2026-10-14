# -*- coding: utf-8 -*-
r"""
Paper 1 — Photo Candidate Selection and Contact-Sheet Generator
===============================================================

Purpose
-------
Automatically select reproducible photographic candidates for the revised
Architecture and Culture results figures.

Selection logic
---------------
1) Canonical reference representative:
   single-lineage reference image with the highest cosine similarity to the
   corresponding DINOv2 reference prototype.

2) Historical / field representative:
   image carrying the target lineage label with the highest cosine similarity
   to the canonical reference prototype.

3) Multi-lineage case:
   image carrying BOTH target labels, ranked by the mean DINOv2 similarity to
   the two corresponding canonical reference prototypes.

The script DOES NOT make the final aesthetic choice for you.
It generates ranked candidates + contact sheets and a manual-selection CSV.
The final plotting script uses the manual-selection CSV.

Dependencies
------------
pip install pandas numpy pillow matplotlib

Inputs
------
<PROJECT_ROOT>/paper1_dinov2_results/
    embeddings_reference.npy
    embeddings_historical.npy
    embeddings_field.npy
    01_reference_embedding_meta.csv
    02_historical_embedding_meta.csv
    03_field_embedding_meta.csv

Outputs
-------
<PROJECT_ROOT>/paper1_photo_candidates/
    00_selection_summary.txt
    01_lineage_representative_candidates.csv
    02_multilineage_case_candidates.csv
    03_manual_selection_template.csv
    contact_sheets\...
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import math
import textwrap
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# =============================================================================
# PATHS / SETTINGS
# =============================================================================

ROOT = Path(str(PROJECT_ROOT))
DINO_DIR = ROOT / "paper1_dinov2_results"
STAGE4_DIR = ROOT / "paper1_visual_genealogy_results"

OUT_DIR = ROOT / "paper1_photo_candidates"
CONTACT_DIR = OUT_DIR / "contact_sheets"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_LINEAGE = 6
TOP_N_MULTILABEL = 8
TOP_PAIR_COUNT_PER_DOMAIN = 4
MIN_PAIR_INTERSECTION = 2

STYLE_IDS = [f"{i:02d}" for i in range(1, 13)]
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

# Pairs explicitly needed in Figure 5.
FIG5_PAIRS = [
    ("01", "04"),
    ("03", "05"),
    ("11", "12"),
    ("03", "10"),
]

# Figure 7 triplet cases.
FIG7_LINEAGES = ["08", "02", "12"]

# Unified QA/contact-sheet palette.
BG = "#FBFAF7"
TEXT = "#2B2B2B"
REF = "#68747E"
HIST = "#B5654D"
FIELD = "#2D6F6A"
BORDER = "#D8D2C8"


# =============================================================================
# IO
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def load_domain(meta_name: str, emb_name: str):
    meta = pd.read_csv(
        require(DINO_DIR / meta_name),
        encoding="utf-8-sig",
    )
    emb = np.load(require(DINO_DIR / emb_name))

    if len(meta) != len(emb):
        raise RuntimeError(
            f"Row mismatch for {meta_name}: meta={len(meta)}, embedding={len(emb)}"
        )

    # DINO script already L2-normalized, but normalize again safely.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-12)

    return meta, emb


def y_matrix(meta: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [meta[f"y_{sid}"].astype(float).astype(int).to_numpy()
         for sid in STYLE_IDS]
    )


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im.copy()


def crop_fill(im: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop to fill a fixed rectangle."""
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
    ncols: int = 6,
    cell_w: int = 360,
    cell_h: int = 250,
):
    if records.empty:
        return

    n = len(records)
    nrows = math.ceil(n / ncols)

    top_h = 72
    text_h = 86
    pad = 14

    canvas_w = pad + ncols * (cell_w + pad)
    canvas_h = top_h + nrows * (cell_h + text_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    # Use default font for maximum portability.
    font_title = ImageFont.load_default()
    font_text = ImageFont.load_default()

    draw.text(
        (pad, 18),
        title,
        fill=TEXT,
        font=font_title,
    )

    domain_color = {
        "reference": REF,
        "historical": HIST,
        "field": FIELD,
    }

    for idx, (_, r) in enumerate(records.iterrows()):
        rr = idx // ncols
        cc = idx % ncols

        x0 = pad + cc * (cell_w + pad)
        y0 = top_h + rr * (cell_h + text_h + pad)

        try:
            im = crop_fill(open_rgb(r["image_path"]), cell_w, cell_h)
            canvas.paste(im, (x0, y0))
        except Exception as e:
            draw.rectangle(
                [x0, y0, x0 + cell_w, y0 + cell_h],
                outline="#999999",
                width=2,
            )
            draw.text(
                (x0 + 8, y0 + 8),
                f"IMAGE ERROR\n{e}",
                fill="#8A2D2D",
                font=font_text,
            )

        color = domain_color.get(str(r.get("domain", "")), REF)
        draw.rectangle(
            [x0, y0, x0 + cell_w, y0 + cell_h],
            outline=color,
            width=5,
        )

        lines = []

        # Robustly handle the two candidate-table schemas:
        # 1) lineage representatives -> "similarity"
        # 2) multi-lineage cases     -> "pair_mean_similarity"
        rank_value = r.get("rank", np.nan)
        if pd.notna(rank_value):
            rank_text = f"Rank {int(rank_value)}"

            sim_value = r.get("similarity", np.nan)
            pair_sim_value = r.get("pair_mean_similarity", np.nan)

            if pd.notna(sim_value):
                rank_text += f" | similarity={float(sim_value):.3f}"
            elif pd.notna(pair_sim_value):
                rank_text += f" | mean similarity={float(pair_sim_value):.3f}"

            lines.append(rank_text)

        pair_value = r.get("pair", np.nan)

        if pd.notna(pair_value) and str(pair_value).strip():
            pair_mean = r.get("pair_mean_similarity", np.nan)

            if pd.notna(pair_mean):
                lines.append(
                    f"{r.get('domain', '')} | pair {pair_value} | "
                    f"mean={float(pair_mean):.3f}"
                )
            else:
                lines.append(
                    f"{r.get('domain', '')} | pair {pair_value}"
                )
        else:
            lines.append(
                f"{r.get('domain', '')} | "
                f"{r.get('style_id', '')} {r.get('style_name', '')}"
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
# PROTOTYPES
# =============================================================================

def build_reference_prototypes(
    ref_meta: pd.DataFrame,
    ref_emb: np.ndarray,
) -> np.ndarray:
    Y = y_matrix(ref_meta)
    single = Y.sum(axis=1) == 1

    protos = []

    for j, sid in enumerate(STYLE_IDS):
        m = single & (Y[:, j] == 1)

        if m.sum() < 5:
            m = Y[:, j] == 1

        p = ref_emb[m].mean(axis=0)
        p = p / max(np.linalg.norm(p), 1e-12)
        protos.append(p)

    return np.stack(protos, axis=0)


# =============================================================================
# CANDIDATE SELECTION
# =============================================================================

def select_lineage_candidates(
    meta: pd.DataFrame,
    emb: np.ndarray,
    prototypes: np.ndarray,
    domain: str,
    top_n: int = TOP_N_LINEAGE,
) -> pd.DataFrame:
    Y = y_matrix(meta)
    rows = []

    for j, sid in enumerate(STYLE_IDS):
        # Reference representatives should be canonical single-lineage examples.
        if domain == "reference":
            m = (Y[:, j] == 1) & (Y.sum(axis=1) == 1)
        else:
            m = Y[:, j] == 1

        idx = np.where(m)[0]
        sims = emb[idx] @ prototypes[j]

        order = np.argsort(-sims)[:top_n]

        for rank, oi in enumerate(order, 1):
            row_idx = idx[oi]
            rows.append({
                "domain": domain,
                "style_id": sid,
                "style_name": STYLE_NAMES[sid],
                "rank": rank,
                "similarity": float(sims[oi]),
                "pixel_sha256": meta.iloc[row_idx]["pixel_sha256"],
                "style_ids": meta.iloc[row_idx].get("style_ids", ""),
                "style_names": meta.iloc[row_idx].get("style_names", ""),
                "multi_lineage": meta.iloc[row_idx].get("multi_lineage", ""),
                "image_path": meta.iloc[row_idx]["image_path"],
            })

    return pd.DataFrame(rows)


def pair_stats(meta: pd.DataFrame) -> pd.DataFrame:
    Y = y_matrix(meta)
    rows = []

    for i, j in combinations(range(12), 2):
        a = Y[:, i].astype(bool)
        b = Y[:, j].astype(bool)

        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        jac = inter / union if union else 0.0

        rows.append({
            "style_id_1": STYLE_IDS[i],
            "style_id_2": STYLE_IDS[j],
            "pair": f"{STYLE_IDS[i]}-{STYLE_IDS[j]}",
            "intersection_n": inter,
            "union_n": union,
            "jaccard": jac,
        })

    return pd.DataFrame(rows)


def select_multilineage_candidates(
    meta: pd.DataFrame,
    emb: np.ndarray,
    prototypes: np.ndarray,
    domain: str,
    selected_pairs: List[Tuple[str, str]],
    top_n: int = TOP_N_MULTILABEL,
) -> pd.DataFrame:
    Y = y_matrix(meta)
    rows = []

    for sid_a, sid_b in selected_pairs:
        ia = STYLE_IDS.index(sid_a)
        ib = STYLE_IDS.index(sid_b)

        m = (Y[:, ia] == 1) & (Y[:, ib] == 1)
        idx = np.where(m)[0]

        if len(idx) == 0:
            continue

        sim_a = emb[idx] @ prototypes[ia]
        sim_b = emb[idx] @ prototypes[ib]
        mean_sim = (sim_a + sim_b) / 2.0
        min_sim = np.minimum(sim_a, sim_b)

        # Primary ranking = mean similarity; secondary = balanced closeness.
        order = np.lexsort((-min_sim, -mean_sim))[:top_n]

        for rank, oi in enumerate(order, 1):
            row_idx = idx[oi]

            rows.append({
                "domain": domain,
                "pair": f"{sid_a}-{sid_b}",
                "style_id_1": sid_a,
                "style_1": STYLE_NAMES[sid_a],
                "style_id_2": sid_b,
                "style_2": STYLE_NAMES[sid_b],
                "rank": rank,
                "sim_style_1": float(sim_a[oi]),
                "sim_style_2": float(sim_b[oi]),
                "pair_mean_similarity": float(mean_sim[oi]),
                "pair_min_similarity": float(min_sim[oi]),
                "pixel_sha256": meta.iloc[row_idx]["pixel_sha256"],
                "style_ids": meta.iloc[row_idx].get("style_ids", ""),
                "style_names": meta.iloc[row_idx].get("style_names", ""),
                "image_path": meta.iloc[row_idx]["image_path"],
            })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("PAPER 1 — PHOTO CANDIDATE SELECTION")
    print("=" * 100)

    ref_meta, ref_emb = load_domain(
        "01_reference_embedding_meta.csv",
        "embeddings_reference.npy",
    )
    hist_meta, hist_emb = load_domain(
        "02_historical_embedding_meta.csv",
        "embeddings_historical.npy",
    )
    field_meta, field_emb = load_domain(
        "03_field_embedding_meta.csv",
        "embeddings_field.npy",
    )

    prototypes = build_reference_prototypes(ref_meta, ref_emb)

    # -------------------------------------------------------------------------
    # A. lineage representatives
    # -------------------------------------------------------------------------
    lineage = pd.concat([
        select_lineage_candidates(
            ref_meta, ref_emb, prototypes, "reference"
        ),
        select_lineage_candidates(
            hist_meta, hist_emb, prototypes, "historical"
        ),
        select_lineage_candidates(
            field_meta, field_emb, prototypes, "field"
        ),
    ], ignore_index=True)

    lineage.to_csv(
        OUT_DIR / "01_lineage_representative_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # B. choose informative multi-lineage pairs from each target domain
    # -------------------------------------------------------------------------
    pair_sets: Dict[str, List[Tuple[str, str]]] = {}

    for domain, meta in [
        ("historical", hist_meta),
        ("field", field_meta),
    ]:
        stats = pair_stats(meta)
        stats.to_csv(
            OUT_DIR / f"pair_stats_{domain}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        good = stats[
            stats["intersection_n"].ge(MIN_PAIR_INTERSECTION)
        ].sort_values(
            ["jaccard", "intersection_n"],
            ascending=[False, False],
        )

        top_pairs = [
            (r["style_id_1"], r["style_id_2"])
            for _, r in good.head(TOP_PAIR_COUNT_PER_DOMAIN).iterrows()
        ]

        # Add theoretically important pairs when they actually occur.
        for p in FIG5_PAIRS:
            row = stats[
                (stats["style_id_1"].eq(p[0]))
                & (stats["style_id_2"].eq(p[1]))
            ]
            if len(row) and int(row.iloc[0]["intersection_n"]) > 0:
                if p not in top_pairs:
                    top_pairs.append(p)

        pair_sets[domain] = top_pairs

    multi = pd.concat([
        select_multilineage_candidates(
            hist_meta,
            hist_emb,
            prototypes,
            "historical",
            pair_sets["historical"],
        ),
        select_multilineage_candidates(
            field_meta,
            field_emb,
            prototypes,
            "field",
            pair_sets["field"],
        ),
    ], ignore_index=True)

    multi.to_csv(
        OUT_DIR / "02_multilineage_case_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # C. contact sheets — lineage representatives
    # -------------------------------------------------------------------------
    for sid in STYLE_IDS:
        dd = lineage[
            lineage["style_id"].eq(sid)
        ].copy()

        # domain order
        dd["domain_order"] = dd["domain"].map({
            "reference": 0,
            "historical": 1,
            "field": 2,
        })
        dd = dd.sort_values(
            ["domain_order", "rank"]
        )

        make_contact_sheet(
            dd,
            CONTACT_DIR / f"Lineage_{sid}_candidates.jpg",
            title=f"{sid} {STYLE_NAMES[sid]} — Reference / Historical / Field candidates",
            ncols=TOP_N_LINEAGE,
        )

    # -------------------------------------------------------------------------
    # D. contact sheets — multi-lineage candidates
    # -------------------------------------------------------------------------
    for (domain, pair), dd in multi.groupby(
        ["domain", "pair"]
    ):
        make_contact_sheet(
            dd.sort_values("rank"),
            CONTACT_DIR / f"Multilineage_{domain}_{pair}.jpg",
            title=f"{domain.capitalize()} multi-lineage case: {pair}",
            ncols=4,
        )

    # -------------------------------------------------------------------------
    # E. manual selection template
    # -------------------------------------------------------------------------
    selection_rows = []

    # One chosen representative per domain × lineage.
    for _, r in lineage.iterrows():
        selection_rows.append({
            "role": "lineage_representative",
            "domain": r["domain"],
            "style_id": r["style_id"],
            "style_name": r["style_name"],
            "pair": "",
            "candidate_rank": int(r["rank"]),
            "similarity": r["similarity"],
            "pair_mean_similarity": "",
            "image_path": r["image_path"],
            "selected": 1 if int(r["rank"]) == 1 else 0,
            "notes": "",
        })

    # One chosen case per domain × pair.
    for _, r in multi.iterrows():
        selection_rows.append({
            "role": "multilineage_case",
            "domain": r["domain"],
            "style_id": "",
            "style_name": "",
            "pair": r["pair"],
            "candidate_rank": int(r["rank"]),
            "similarity": "",
            "pair_mean_similarity": r["pair_mean_similarity"],
            "image_path": r["image_path"],
            "selected": 1 if int(r["rank"]) == 1 else 0,
            "notes": "",
        })

    sel = pd.DataFrame(selection_rows)
    sel.to_csv(
        OUT_DIR / "03_manual_selection_template.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Helpful subset listing the Figure-5 / Figure-7 required cases.
    fig_req_rows = []

    for a, b in FIG5_PAIRS:
        fig_req_rows.append({
            "figure": "Figure 5",
            "requirement": f"Reference representatives for {a} and {b}",
        })

    for sid in FIG7_LINEAGES:
        fig_req_rows.append({
            "figure": "Figure 7",
            "requirement": f"Reference + Historical + Field representatives for {sid}",
        })

    pd.DataFrame(fig_req_rows).to_csv(
        OUT_DIR / "04_required_photo_roles.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = f"""
PHOTO CANDIDATE SELECTION COMPLETE

Reference images:  {len(ref_meta)}
Historical images: {len(hist_meta)}
Field images:      {len(field_meta)}

Lineage candidate rows:     {len(lineage)}
Multi-lineage candidate rows:{len(multi)}

NEXT STEP
---------
1. Open:
   {OUT_DIR / '03_manual_selection_template.csv'}

2. Review contact sheets in:
   {CONTACT_DIR}

3. For each domain × lineage and domain × pair, keep exactly ONE row selected=1.
   You may simply keep the default rank-1 candidate when visually acceptable.

4. Save the reviewed file as:
   {OUT_DIR / '03_manual_selection.csv'}

5. Run the V2 final plotting script.
"""

    (OUT_DIR / "00_selection_summary.txt").write_text(
        summary.strip() + "\n",
        encoding="utf-8",
    )

    print(summary)


if __name__ == "__main__":
    main()
