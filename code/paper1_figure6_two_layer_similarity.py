# -*- coding: utf-8 -*-
r"""
Paper 1 — Supplementary Figure S5
Complete Pairwise Similarity Structures of the Two-Layer Visual Genealogy
=========================================================================

Creates two side-by-side 12 × 12 heatmaps:
(a) DINOv2 reference-prototype similarity = whole-image visual syntax
(b) A–D architectural-symbol similarity = component-level vocabulary

IMPORTANT
---------
- The A–D matrix uses ONLY the 69 architectural morphology symbols.
- Canonical A–D profiles are estimated ONLY from single-lineage reference images.
- No figure caption is embedded inside the figure.
- Output includes PNG, PDF, and the derived A–D similarity matrix CSV.

Dependencies
------------
pip install pandas numpy matplotlib scikit-learn

Run
---
python paper1_figure6_two_layer_similarity.py
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(str(PROJECT_ROOT))

DINO_DIR = ROOT / "paper1_dinov2_results"
FROZEN_DIR = ROOT / "paper1_frozen_dataset_v3"
OUT_DIR = ROOT / "paper1_final_figures_v4"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DINO_MATRIX_FILE = DINO_DIR / "04_prototype_similarity_reference.csv"
MASTER_FILE = FROZEN_DIR / "01_unique_image_multilabel_master.csv"

# =============================================================================
# SETTINGS
# =============================================================================

DPI = 600
ALPHA = 0.5

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

STYLE_YLABELS = [
    f"{sid}  {STYLE_NAMES[sid]}"
    for sid in STYLE_IDS
]

AD_SYMBOLS = (
    [f"A{i}" for i in range(1, 13)]
    + [f"B{i}" for i in range(1, 17)]
    + [f"C{i}" for i in range(1, 21)]
    + [f"D{i}" for i in range(1, 22)]
)

# Unified Architecture & Culture palette
BG = "#FBFAF7"
TEXT = "#262626"
GRID = "#E1DDD5"
NAVY = "#35506B"
OCHRE = "#B28A45"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# =============================================================================
# HELPERS
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def parse_symbol_counts(value) -> Dict[str, int]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        d = json.loads(value)
        return {
            str(k): int(v)
            for k, v in d.items()
            if int(v) > 0
        }
    except Exception:
        return {}


def symbol_vector(value, symbols: List[str]) -> np.ndarray:
    d = parse_symbol_counts(value)
    return np.array(
        [d.get(sym, 0) for sym in symbols],
        dtype=float,
    )


def y_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df[f"y_{sid}"].astype(float).astype(int).to_numpy()
        for sid in STYLE_IDS
    ])


def build_ad_similarity(master: pd.DataFrame):
    """
    Canonical component-vocabulary profiles:
    reference domain + single-lineage images only.
    """
    ref = master[
        master["dataset_role"].eq("development_reference")
    ].copy().reset_index(drop=True)

    Y = y_matrix(ref)

    X = np.vstack(
        ref["symbol_counts_json"]
        .map(lambda x: symbol_vector(x, AD_SYMBOLS))
        .to_numpy()
    )

    single = (Y.sum(axis=1) == 1)

    counts = np.zeros((12, len(AD_SYMBOLS)), dtype=float)
    n_images = np.zeros(12, dtype=int)

    for k in range(12):
        m = single & (Y[:, k] == 1)
        n_images[k] = int(m.sum())
        counts[k] = X[m].sum(axis=0)

    profiles = (
        counts + ALPHA
    ) / (
        counts.sum(axis=1, keepdims=True)
        + ALPHA * len(AD_SYMBOLS)
    )

    sim = cosine_similarity(profiles)

    return sim, n_images


def annotate_matrix(ax, M: np.ndarray):
    """
    Annotate all cells with two decimal places.
    Automatically switch text color on darker cells.
    """
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = float(M[i, j])

            # darker blue/orange extremes get white text
            text_color = "white" if v >= 0.82 else TEXT

            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=7.1,
                color=text_color,
            )


def style_heatmap_axis(ax, title: str):
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        STYLE_IDS,
        rotation=0,
        fontsize=9,
    )

    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(
        STYLE_YLABELS,
        fontsize=8.9,
    )

    ax.set_xlabel("Visual lineage ID")
    ax.set_title(title, pad=10)

    # Thin white grid between cells.
    ax.set_xticks(
        np.arange(-0.5, 12, 1),
        minor=True,
    )
    ax.set_yticks(
        np.arange(-0.5, 12, 1),
        minor=True,
    )
    ax.grid(
        which="minor",
        color=BG,
        linewidth=0.8,
    )
    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#8F8A82")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("SUPPLEMENTARY FIGURE S5 — TWO-LAYER SIMILARITY MATRICES")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # DINOv2 matrix
    # -------------------------------------------------------------------------
    dino_df = pd.read_csv(
        require(DINO_MATRIX_FILE),
        encoding="utf-8-sig",
        index_col=0,
    )
    dino = dino_df.to_numpy(dtype=float)

    if dino.shape != (12, 12):
        raise RuntimeError(
            f"Unexpected DINO matrix shape: {dino.shape}"
        )

    # -------------------------------------------------------------------------
    # A–D symbol matrix
    # -------------------------------------------------------------------------
    master = pd.read_csv(
        require(MASTER_FILE),
        encoding="utf-8-sig",
    )

    ad_sim, n_images = build_ad_similarity(master)

    ad_df = pd.DataFrame(
        ad_sim,
        index=STYLE_YLABELS,
        columns=STYLE_IDS,
    )
    ad_csv = OUT_DIR / "Figure_S5_AD_symbol_similarity_matrix.csv"
    ad_df.to_csv(
        ad_csv,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2,
        figsize=(15.8, 7.1),
        gridspec_kw={"wspace": 0.34},
    )

    # Similarity range: 0–1 for direct cross-panel comparability.
    norm = Normalize(vmin=0.0, vmax=1.0)

    # Panel (a): DINO
    im1 = axes[0].imshow(
        dino,
        cmap="Blues",
        norm=norm,
        aspect="equal",
    )
    style_heatmap_axis(
        axes[0],
        "Whole-image visual syntax\n(DINOv2 reference prototypes)",
    )
    annotate_matrix(axes[0], dino)

    axes[0].text(
        -0.13,
        1.04,
        "(a)",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
    )

    # Panel (b): A–D
    im2 = axes[1].imshow(
        ad_sim,
        cmap="YlOrBr",
        norm=norm,
        aspect="equal",
    )
    style_heatmap_axis(
        axes[1],
        "Component-level architectural vocabulary\n(A–D symbol profiles)",
    )
    annotate_matrix(axes[1], ad_sim)

    axes[1].text(
        -0.13,
        1.04,
        "(b)",
        transform=axes[1].transAxes,
        ha="left",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=TEXT,
    )

    # One shared colorbar = easier comparison.
    cbar = fig.colorbar(
        im2,
        ax=axes,
        orientation="vertical",
        fraction=0.022,
        pad=0.025,
        shrink=0.88,
    )
    cbar.set_label(
        "Cosine similarity",
        fontsize=10.5,
    )
    cbar.ax.tick_params(
        labelsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.93,
        bottom=0.10,
        top=0.90,
    )

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    png = OUT_DIR / "Figure_S5_Two_Layer_Similarity_Matrices.png"
    pdf = OUT_DIR / "Figure_S5_Two_Layer_Similarity_Matrices.pdf"

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.07,
        facecolor=BG,
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.07,
        facecolor=BG,
    )
    plt.close(fig)

    # Also save reference counts used for canonical profiles.
    count_df = pd.DataFrame({
        "style_id": STYLE_IDS,
        "style_name": [STYLE_NAMES[s] for s in STYLE_IDS],
        "single_lineage_reference_images": n_images,
    })
    count_df.to_csv(
        OUT_DIR / "Figure_S5_reference_profile_sample_sizes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")
    print(f"[SAVED] {ad_csv}")
    print()
    print("Canonical single-lineage reference sample sizes:")
    print(count_df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
