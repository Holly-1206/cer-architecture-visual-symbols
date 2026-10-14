# -*- coding: utf-8 -*-
r"""
Paper 1 — Final Figure 3–Figure 7 V4 Generator
==============================================

This is the revised publication-quality plotting script for Architecture and Culture.

Major revisions
---------------
1. No figure captions / "Figure X..." text inside figures.
2. Unified muted heritage palette.
3. No default rainbow colors.
4. Overlap reduction through compact titles, external legends, sparse labeling,
   fixed layouts, and optional adjustText.
5. Real architectural photographs are integrated into Figures 4–7.
6. Figure 4 prioritizes A–D architectural morphology; E railway context is
   exported separately as a Supplementary Figure.
7. VHI is not used.

Dependencies
------------
pip install pandas numpy scipy scikit-learn matplotlib networkx pillow adjustText

Run
---
python paper1_figures_3_to_7_v2.py

Manual selection file
---------------------
Preferred:
<PROJECT_ROOT>/paper1_photo_candidates/03_manual_selection.csv

If it does not exist, the script falls back to:
03_manual_selection_template.csv
and uses the default rank-1 selections.

Outputs
-------
<PROJECT_ROOT>/paper1_final_figures_v4/
    Figure_3_v4.png/.pdf
    Figure_4_v4.png/.pdf
    Figure_5_v4.png/.pdf
    Figure_6_v4.png/.pdf
    Figure_7_v4.png/.pdf
    Figure_S4_Railway_Context_v4.png/.pdf
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import json
import argparse
import math
import warnings
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import networkx as nx

from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except Exception:
    HAS_ADJUSTTEXT = False

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 0. PATHS
# =============================================================================

ROOT = Path(str(PROJECT_ROOT))

CONVNEXT_DIR = ROOT / "paper1_model_results" / "convnext_tiny"
DINO_DIR = ROOT / "paper1_dinov2_results"
STAGE4_DIR = ROOT / "paper1_visual_genealogy_results"
FROZEN_DIR = ROOT / "paper1_frozen_dataset_v3"
PHOTO_DIR = ROOT / "paper1_photo_candidates"
DEFAULT_SELECTION_FILE = PHOTO_DIR / "03_manual_selection.csv"

OUT_DIR = ROOT / "paper1_final_figures_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. FINAL PALETTE
# =============================================================================

BG = "#FBFAF7"          # ivory white
TEXT = "#262626"        # charcoal
GRID = "#DDD8CF"        # warm grey
REFERENCE = "#68747E"   # slate
HISTORICAL = "#B5654D"  # muted terracotta
FIELD = "#2D6F6A"       # deep teal
DINO = "#35506B"        # deep navy
SYMBOL = "#B28A45"      # muted ochre
NEUTRAL = "#9B968D"     # warm neutral
LIGHT_GREY = "#C8C3BA"
DARK_GREY = "#5A5956"
ACCENT = "#7A5F58"

HERITAGE_CMAP = LinearSegmentedColormap.from_list(
    "heritage_diverging",
    [DINO, "#F3EFE6", HISTORICAL],
)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DPI = 600
ALPHA = 0.5
PERMUTATIONS = 10000
SEED = 20260917
MIN_SUPPORT = 15
NETWORK_EDGE_THRESHOLD = 0.05

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

STYLE_SHORT = {
    "01": "Russian\nTraditional",
    "02": "Russian\nTimber",
    "03": "Russian\nRailway",
    "04": "Byzantine /\nOrthodox",
    "05": "Eclecticism",
    "06": "Art Nouveau",
    "07": "Baroque",
    "08": "Chinese\nBaroque",
    "09": "Classical /\nNeo.",
    "10": "Gothic",
    "11": "Art Deco",
    "12": "Early\nModernism",
}

STYLE_TICK = {
    sid: f"{sid}  {STYLE_NAMES[sid]}"
    for sid in STYLE_IDS
}

AD_SYMBOLS = (
    [f"A{i}" for i in range(1, 13)]
    + [f"B{i}" for i in range(1, 17)]
    + [f"C{i}" for i in range(1, 21)]
    + [f"D{i}" for i in range(1, 22)]
)

E_SYMBOLS = [f"E{i}" for i in range(1, 16)]

SYMBOL_NAMES = {
    "A1": "Onion dome", "A2": "Tent roof", "A3": "Gable roof",
    "A4": "Hipped roof", "A5": "Multi-slope roof", "A6": "Metal-sheet roof",
    "A7": "Dormer", "A8": "Flat roof", "A9": "Parapet", "A10": "Cross",
    "A11": "Chimney", "A12": "Projecting eave",

    "B1": "Semicircular arch", "B2": "Segmental arch", "B3": "Pointed arch",
    "B4": "Rectangular window", "B5": "Paired window", "B6": "Triple window",
    "B7": "Bay window", "B8": "Small vent", "B9": "Gridded window",
    "B10": "Window surround", "B11": "Keystone", "B12": "Window sill",
    "B13": "Arched door", "B14": "Entrance porch", "B15": "Entrance steps",
    "B16": "Door fanlight",

    "C1": "Red brick", "C2": "Yellow brick", "C3": "Painted brick",
    "C4": "Cement plaster", "C5": "Stone base", "C6": "Stone veneer",
    "C7": "Log wall", "C8": "Vertical timber", "C9": "Rustication",
    "C10": "Quoin", "C11": "Brick–wood structure", "C12": "Wood structure",
    "C13": "Brick–stone structure", "C14": "Brick–concrete structure",
    "C15": "Steel truss", "C16": "Concrete arch", "C17": "Wave arch",
    "C18": "Brick bond", "C19": "Interlocking masonry", "C20": "Pilaster",

    "D1": "Cornice", "D2": "White band", "D3": "Triangular pediment",
    "D4": "Curved pediment", "D5": "Volute", "D6": "Floral relief",
    "D7": "Art Nouveau curve", "D8": "Russian wood lace",
    "D9": "Contrasting trim", "D10": "Cloud motif", "D11": "Hui-pattern motif",
    "D12": "Bat motif", "D13": "Pomegranate motif", "D14": "Grape motif",
    "D15": "Peony motif", "D16": "Chinese knot", "D17": "Coin motif",
    "D18": "Dougong", "D19": "Balustrade", "D20": "Ironwork", "D21": "Plaque",

    "E1": "Station canopy", "E2": "Platform", "E3": "Rail track",
    "E4": "Roundhouse", "E5": "Turntable", "E6": "Engine-shed door",
    "E7": "Water tower", "E8": "Warehouse door", "E9": "Industrial chimney",
    "E10": "Bridge truss", "E11": "Stone-arch bridge",
    "E12": "Tunnel portal", "E13": "Culvert",
    "E14": "Railway signal", "E15": "Railway fence",
}

FIG5_PAIRS = [
    ("01", "04"),
    ("03", "05"),
    ("11", "12"),
    ("03", "10"),
]

FIG7_LINEAGES = ["08", "02", "12"]

# Main-text photographic exemplars are illustrative, whereas the quantitative
# canonical A–D profiles are still estimated only from single-lineage
# reference images. Russian Railway (03) uses a visually reviewed field
# building because its reference candidates are dominated by rolling stock.
EXEMPLAR_DOMAIN = {sid: "reference" for sid in STYLE_IDS}
EXEMPLAR_DOMAIN["03"] = "field"

# Fixed, visually reviewed multi-lineage cases for Figure 6.
FIG6_CASES = [
    ("historical", "01-04"),
    ("historical", "03-05"),
    ("field", "01-02"),
    ("field", "07-08"),
]

# =============================================================================
# 2. UTILITIES
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(require(path), encoding="utf-8-sig", **kwargs)


def save_figure(fig: plt.Figure, stem: str):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor=BG,
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor=BG,
    )
    plt.close(fig)

    print(f"[SAVED] {png}")
    print(f"[SAVED] {pdf}")


def clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)


def panel_label(ax, label: str):
    ax.text(
        0.0, 1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
    )


def style_id_clean(x) -> str:
    try:
        return f"{int(float(x)):02d}"
    except Exception:
        return str(x).strip().zfill(2)


def parse_counts(value) -> Dict[str, int]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        d = json.loads(value)
        return {str(k): int(v) for k, v in d.items() if int(v) > 0}
    except Exception:
        return {}


def symbol_vector(value, symbols: List[str]) -> np.ndarray:
    d = parse_counts(value)
    return np.array([d.get(s, 0) for s in symbols], dtype=float)


def y_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        df[f"y_{sid}"].astype(float).astype(int).to_numpy()
        for sid in STYLE_IDS
    ])


def load_master() -> pd.DataFrame:
    df = read_csv(FROZEN_DIR / "01_unique_image_multilabel_master.csv")

    valid_roles = {
        "development_reference",
        "external_historical",
        "external_field",
    }

    df = df[df["dataset_role"].isin(valid_roles)].copy()

    role_map = {
        "development_reference": "reference",
        "external_historical": "historical",
        "external_field": "field",
    }

    df["domain"] = df["dataset_role"].map(role_map)
    return df.reset_index(drop=True)


def build_profiles(
    master: pd.DataFrame,
    symbols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    Y = y_matrix(master)

    X = np.vstack(
        master["symbol_counts_json"]
        .map(lambda x: symbol_vector(x, symbols))
        .to_numpy()
    )

    ref = master["domain"].eq("reference").to_numpy()
    single = Y.sum(axis=1) == 1
    canonical = ref & single

    counts = np.zeros((12, len(symbols)), dtype=float)

    for k in range(12):
        m = canonical & (Y[:, k] == 1)
        counts[k] = X[m].sum(axis=0)

    profiles = (
        counts + ALPHA
    ) / (
        counts.sum(axis=1, keepdims=True)
        + ALPHA * len(symbols)
    )

    return counts, profiles


def log_enrichment(counts: np.ndarray) -> np.ndarray:
    profiles = (
        counts + ALPHA
    ) / (
        counts.sum(axis=1, keepdims=True)
        + ALPHA * counts.shape[1]
    )

    global_counts = counts.sum(axis=0)
    global_profile = (
        global_counts + ALPHA
    ) / (
        global_counts.sum()
        + ALPHA * counts.shape[1]
    )

    return np.log(profiles / global_profile[None, :])


def upper_triangle(M):
    return M[np.triu_indices_from(M, k=1)]


def permutation_spearman(A, B, permutations=PERMUTATIONS, seed=SEED):
    rng = np.random.default_rng(seed)

    a = upper_triangle(A)
    b = upper_triangle(B)
    rho = spearmanr(a, b).statistic

    hits = 0
    n = len(B)

    for _ in range(permutations):
        p = rng.permutation(n)
        bp = upper_triangle(B[np.ix_(p, p)])
        rp = spearmanr(a, bp).statistic
        if abs(rp) >= abs(rho):
            hits += 1

    pval = (hits + 1) / (permutations + 1)
    return float(rho), float(pval)


def load_square(path: Path) -> pd.DataFrame:
    return read_csv(path, index_col=0).astype(float)


def open_rgb(path: str):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im.copy()


def crop_fill(im: Image.Image, ratio=4/3):
    w, h = im.size
    current = w / h

    if current > ratio:
        new_w = int(h * ratio)
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / ratio)
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))

    return im


def show_photo(ax, path: str, title=None, subtitle=None, border_color=None):
    try:
        im = crop_fill(open_rgb(path))
        ax.imshow(im)
    except Exception:
        ax.set_facecolor("#EFECE6")
        ax.text(
            0.5, 0.5,
            "Image unavailable",
            ha="center", va="center",
            transform=ax.transAxes,
            color="#8A2D2D",
        )

    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.2)
        sp.set_color(border_color or LIGHT_GREY)

    if title:
        ax.set_title(
            title,
            fontsize=9.2,
            pad=5,
            color=TEXT,
        )

    if subtitle:
        ax.text(
            0.5, -0.055,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.0,
            color=DARK_GREY,
        )


# =============================================================================
# 3. MANUAL PHOTO SELECTION
# =============================================================================

def load_selection(selection_file: Path | None = None) -> pd.DataFrame:
    """
    Load the reviewed photo-selection CSV.

    Search order:
    1) --selection-file supplied on the command line
    2) default project data location
    3) same directory as this Python script
    4) current working directory

    This avoids failing merely because the CSV was downloaded next to the script
    rather than copied into the data folder.
    """
    candidates = []

    if selection_file is not None:
        candidates.append(Path(selection_file))

    candidates.extend([
        DEFAULT_SELECTION_FILE,
        Path(__file__).resolve().parent / "03_manual_selection.csv",
        Path.cwd() / "03_manual_selection.csv",
    ])

    # Deduplicate while preserving order
    checked = []
    for p in candidates:
        p = Path(p)
        if p not in checked:
            checked.append(p)

    chosen = None
    for p in checked:
        if p.exists():
            chosen = p
            break

    if chosen is None:
        paths = "\n".join(f"  - {p}" for p in checked)
        raise FileNotFoundError(
            "03_manual_selection.csv was not found.\n"
            "Checked these locations:\n"
            f"{paths}\n\n"
            "Either copy the CSV to the default data folder, place it next to "
            "this .py file, or pass:\n"
            '  --selection-file "FULL_PATH_TO_03_manual_selection.csv"'
        )

    print(f"[INFO] Using photo selection file: {chosen}")

    df = pd.read_csv(chosen, encoding="utf-8-sig")
    df["selected"] = pd.to_numeric(
        df["selected"], errors="coerce"
    ).fillna(0).astype(int)
    return df


def get_lineage_record(
    selection: pd.DataFrame,
    domain: str,
    sid: str,
) -> pd.Series:
    q = selection[
        selection["role"].eq("lineage_representative")
        & selection["domain"].eq(domain)
        & selection["style_id"].astype(str).map(style_id_clean).eq(sid)
        & selection["selected"].eq(1)
    ].copy()

    if len(q) != 1:
        raise RuntimeError(
            f"Expected exactly one selected lineage photo for "
            f"{domain}/{sid}, found {len(q)}."
        )

    return q.iloc[0]


def get_lineage_photo(
    selection: pd.DataFrame,
    domain: str,
    sid: str,
) -> str:
    return str(
        get_lineage_record(
            selection, domain, sid
        )["image_path"]
    )


def get_exemplar_record(
    selection: pd.DataFrame,
    sid: str,
) -> pd.Series:
    domain = EXEMPLAR_DOMAIN[sid]
    return get_lineage_record(
        selection,
        domain,
        sid,
    )


def get_pair_record(
    selection: pd.DataFrame,
    domain: str,
    pair: str,
) -> pd.Series:
    q = selection[
        selection["role"].eq("multilineage_case")
        & selection["domain"].eq(domain)
        & selection["pair"].astype(str).eq(pair)
        & selection["selected"].eq(1)
    ].copy()

    if len(q) != 1:
        raise RuntimeError(
            f"Expected exactly one selected multi-lineage case for "
            f"{domain}/{pair}, found {len(q)}."
        )

    return q.iloc[0]


def get_pair_photo(
    selection: pd.DataFrame,
    domain: str,
    pair: str,
) -> str:
    return str(
        get_pair_record(
            selection, domain, pair
        )["image_path"]
    )


def domain_color(domain: str) -> str:
    return {
        "reference": REFERENCE,
        "historical": HISTORICAL,
        "field": FIELD,
    }[domain]




# =============================================================================
# 4. FIGURE 3 V2
# =============================================================================

def make_figure_3():
    cv = read_csv(CONVNEXT_DIR / "cv_fold_metrics.csv")

    with open(require(CONVNEXT_DIR / "final_metrics.json"), "r", encoding="utf-8") as f:
        conv = json.load(f)

    dino_proto = read_csv(DINO_DIR / "06_prototype_external_metrics.csv")
    dino_probe = read_csv(DINO_DIR / "07_linear_probe_external_metrics.csv")
    dino_lineage = read_csv(DINO_DIR / "08_per_lineage_external_metrics.csv")
    dino_lineage["style_id"] = dino_lineage["style_id"].map(style_id_clean)

    conv_h = read_csv(
        CONVNEXT_DIR / "historical_external_per_label_metrics.csv",
        dtype={"style_id": str},
    )
    conv_f = read_csv(
        CONVNEXT_DIR / "field_external_per_label_metrics.csv",
        dtype={"style_id": str},
    )

    conv_h["style_id"] = conv_h["style_id"].map(style_id_clean)
    conv_f["style_id"] = conv_f["style_id"].map(style_id_clean)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.8, 5.8),
        gridspec_kw={"width_ratios": [1.0, 1.05, 1.15]},
    )

    # (a)
    ax = axes[0]
    ax.plot(
        cv["fold"], cv["macro_f1"],
        marker="o", color=DINO, linewidth=1.8,
        label="Macro-F1",
    )
    ax.plot(
        cv["fold"], cv["mAP"],
        marker="s", color=SYMBOL, linewidth=1.8,
        label="mAP",
    )
    ax.plot(
        cv["fold"], cv["macro_AUROC"],
        marker="^", color=FIELD, linewidth=1.8,
        label="Macro-AUROC",
    )

    ax.set_xticks(cv["fold"])
    ax.set_ylim(0.45, 0.90)
    ax.set_xlabel("Cross-validation fold")
    ax.set_ylabel("Score")
    ax.set_title("Development-set stability")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    clean_spines(ax)
    panel_label(ax, "(a)")
    ax.legend(
        frameon=False,
        ncol=1,
        loc="lower right",
    )

    # (b)
    ax = axes[1]

    rows = []
    for domain in ["historical", "field"]:
        rows.append({
            "method": "ConvNeXt",
            "domain": domain,
            "AUROC": conv[f"{domain}_macro_AUROC"],
            "mAP": conv[f"{domain}_mAP"],
        })

    for _, r in dino_proto.iterrows():
        rows.append({
            "method": "DINOv2 prototype",
            "domain": r["domain"],
            "AUROC": r["macro_AUROC"],
            "mAP": r["mAP"],
        })

    for _, r in dino_probe.iterrows():
        rows.append({
            "method": "DINOv2 linear probe",
            "domain": r["domain"],
            "AUROC": r["macro_AUROC"],
            "mAP": r["mAP"],
        })

    perf = pd.DataFrame(rows)

    cats = [
        ("historical", "AUROC", "Historical\nAUROC"),
        ("historical", "mAP", "Historical\nmAP"),
        ("field", "AUROC", "Field\nAUROC"),
        ("field", "mAP", "Field\nmAP"),
    ]

    methods = [
        ("ConvNeXt", REFERENCE),
        ("DINOv2 prototype", HISTORICAL),
        ("DINOv2 linear probe", FIELD),
    ]

    x = np.arange(len(cats))
    width = 0.23

    for i, (method, color) in enumerate(methods):
        vals = []
        for domain, metric, _ in cats:
            vals.append(
                perf[
                    perf["method"].eq(method)
                    & perf["domain"].eq(domain)
                ].iloc[0][metric]
            )

        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            color=color,
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in cats])
    ax.set_ylim(0, 0.80)
    ax.set_ylabel("Score")
    ax.set_title("Cross-domain transfer")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    clean_spines(ax)
    panel_label(ax, "(b)")
    ax.legend(
        frameon=False,
        loc="upper right",
    )

    # (c)
    ax = axes[2]

    dino_p = dino_lineage[dino_lineage["method"].eq("prototype")]

    texts = []

    for domain, conv_df, marker, color in [
        ("historical", conv_h, "o", HISTORICAL),
        ("field", conv_f, "^", FIELD),
    ]:
        dd = dino_p[dino_p["domain"].eq(domain)]

        m = conv_df[
            ["style_id", "AUROC"]
        ].merge(
            dd[["style_id", "AUROC"]],
            on="style_id",
            suffixes=("_convnext", "_dino"),
        )

        ax.scatter(
            m["AUROC_convnext"],
            m["AUROC_dino"],
            s=58,
            marker=marker,
            color=color,
            alpha=0.88,
            edgecolor=BG,
            linewidth=0.6,
            label=domain.capitalize(),
        )

        m["delta"] = m["AUROC_dino"] - m["AUROC_convnext"]

        label_ids = set(
            m.nlargest(2, "delta")["style_id"].tolist()
            + m.nsmallest(1, "delta")["style_id"].tolist()
        )

        label_ids |= {"03", "08", "12"}

        for _, r in m.iterrows():
            if r["style_id"] in label_ids:
                t = ax.text(
                    r["AUROC_convnext"],
                    r["AUROC_dino"],
                    r["style_id"],
                    fontsize=8,
                    color=TEXT,
                )
                texts.append(t)

    lo, hi = 0.30, 1.00
    ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--",
        color=NEUTRAL,
        linewidth=1.0,
    )

    if HAS_ADJUSTTEXT and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="-",
                color=LIGHT_GREY,
                lw=0.6,
            ),
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ConvNeXt external AUROC")
    ax.set_ylabel("DINOv2 prototype external AUROC")
    ax.set_title("Lineage-specific transfer")
    ax.grid(color=GRID, linewidth=0.7)
    clean_spines(ax)
    panel_label(ax, "(c)")
    ax.legend(frameon=False, loc="lower right")

    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.18,
        top=0.88,
        wspace=0.34,
    )

    save_figure(fig, "Figure_3_v4")


# =============================================================================
# 5. FIGURE 4 V2 — PHOTO ATLAS + MORPHOLOGY SIGNATURES
# =============================================================================

def top_ad_signatures(master: pd.DataFrame, top_k=4):
    counts, profiles = build_profiles(master, AD_SYMBOLS)
    enrich = log_enrichment(counts)

    rows = []
    matrix = np.zeros((12, top_k))
    codes = np.empty((12, top_k), dtype=object)

    for k, sid in enumerate(STYLE_IDS):
        candidates = [
            j for j in range(len(AD_SYMBOLS))
            if counts[k, j] >= 5
        ]
        if len(candidates) < top_k:
            candidates = list(range(len(AD_SYMBOLS)))

        top = sorted(
            candidates,
            key=lambda j: enrich[k, j],
            reverse=True,
        )[:top_k]

        for rank, j in enumerate(top):
            code = AD_SYMBOLS[j]
            matrix[k, rank] = enrich[k, j]
            codes[k, rank] = code

            rows.append({
                "style_id": sid,
                "style_name": STYLE_NAMES[sid],
                "rank": rank + 1,
                "symbol_code": code,
                "symbol_name": SYMBOL_NAMES[code],
                "log_enrichment": enrich[k, j],
                "raw_count": int(counts[k, j]),
            })

    return pd.DataFrame(rows), matrix, codes


def make_figure_4(master: pd.DataFrame, selection: pd.DataFrame):
    sig, matrix, codes = top_ad_signatures(master, top_k=4)

    # V4: the atlas and signature matrix are vertically separated so that
    # photograph labels never collide with heatmap lineage labels.
    fig = plt.figure(figsize=(16.5, 13.2))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.80, 1.0],
        hspace=0.18,
    )

    # -----------------------------------------------------------------
    # (a) 12-photo atlas
    # -----------------------------------------------------------------
    atlas_gs = outer[0, 0].subgridspec(
        3, 4,
        wspace=0.055,
        hspace=0.16,
    )

    atlas_axes = []

    for idx, sid in enumerate(STYLE_IDS):
        ax = fig.add_subplot(
            atlas_gs[idx // 4, idx % 4]
        )
        atlas_axes.append(ax)

        exemplar = get_exemplar_record(
            selection,
            sid,
        )
        path = str(exemplar["image_path"])
        exemplar_domain = str(exemplar["domain"])

        s2 = (
            sig[sig["style_id"].eq(sid)]
            .sort_values("rank")
            .head(2)
        )

        subtitle = " · ".join(
            f"{r.symbol_code} {r.symbol_name}"
            for _, r in s2.iterrows()
        )

        # Keep all supporting text INSIDE the image area to avoid vertical
        # overlap between rows.
        show_photo(
            ax,
            path,
            title=f"{sid}  {STYLE_NAMES[sid]}",
            subtitle=None,
            border_color=domain_color(exemplar_domain),
        )

        ax.text(
            0.02, 0.035,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            color=TEXT,
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor=BG,
                edgecolor="none",
                alpha=0.88,
            ),
        )

        ax.text(
            0.97, 0.965,
            exemplar_domain.capitalize(),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor=domain_color(exemplar_domain),
                edgecolor="none",
                alpha=0.92,
            ),
        )

    panel_label(atlas_axes[0], "(a)")

    # -----------------------------------------------------------------
    # (b) compact signature matrix, centered with generous margins
    # -----------------------------------------------------------------
    heat_gs = outer[1, 0].subgridspec(
        1, 3,
        width_ratios=[0.12, 0.76, 0.12],
        wspace=0.0,
    )
    ax = fig.add_subplot(heat_gs[0, 1])

    vmin = float(np.nanmin(matrix))
    vmax = float(np.nanmax(matrix))
    norm = TwoSlopeNorm(
        vmin=min(vmin, -0.01),
        vcenter=0,
        vmax=max(vmax, 0.01),
    )

    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=HERITAGE_CMAP,
        norm=norm,
    )

    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(
        [STYLE_TICK[s] for s in STYLE_IDS],
        fontsize=9.6,
    )
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(
        ["1st", "2nd", "3rd", "4th"],
        fontsize=9.6,
    )

    for i in range(12):
        for j in range(4):
            code = codes[i, j]
            name = SYMBOL_NAMES[code]
            if len(name) > 20:
                name = name[:18] + "…"

            ax.text(
                j, i,
                f"{code}\n{name}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=TEXT,
            )

    ax.set_title(
        "Top architectural morphology signatures (A–D symbols)",
        fontsize=12.5,
        pad=10,
    )
    panel_label(ax, "(b)")

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.025,
        pad=0.018,
    )
    cbar.set_label("Log enrichment", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)

    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.045,
        top=0.965,
    )

    save_figure(fig, "Figure_4_v4")


def make_supplementary_railway_context(master: pd.DataFrame):
    counts, profiles = build_profiles(master, E_SYMBOLS)
    enrich = log_enrichment(counts)

    fig, ax = plt.subplots(figsize=(12.2, 7.0))

    vmin = float(np.nanmin(enrich))
    vmax = float(np.nanmax(enrich))

    im = ax.imshow(
        enrich,
        aspect="auto",
        cmap=HERITAGE_CMAP,
        norm=TwoSlopeNorm(
            vmin=min(vmin, -0.01),
            vcenter=0,
            vmax=max(vmax, 0.01),
        ),
    )

    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(
        [STYLE_TICK[s] for s in STYLE_IDS],
        fontsize=9.8,
    )
    ax.set_xticks(np.arange(15))
    ax.set_xticklabels(
        [f"E{i}" for i in range(1, 16)],
        rotation=45,
        ha="right",
        fontsize=9.5,
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.02,
    )
    cbar.set_label("Log enrichment", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)

    # Two-line definition block instead of one very long xlabel.
    line1 = (
        "E1 canopy · E2 platform · E3 track · E4 roundhouse · "
        "E5 turntable · E6 engine-shed door · E7 water tower · "
        "E8 warehouse door"
    )
    line2 = (
        "E9 chimney · E10 bridge truss · E11 stone-arch bridge · "
        "E12 tunnel portal · E13 culvert · E14 signal · E15 fence"
    )

    fig.text(
        0.52, 0.070,
        line1,
        ha="center",
        va="center",
        fontsize=8.8,
        color=DARK_GREY,
    )
    fig.text(
        0.52, 0.040,
        line2,
        ha="center",
        va="center",
        fontsize=8.8,
        color=DARK_GREY,
    )

    fig.subplots_adjust(
        left=0.22,
        right=0.94,
        bottom=0.17,
        top=0.96,
    )

    save_figure(fig, "Figure_S4_Railway_Context_v4")


# =============================================================================
# 6. FIGURE 5 V2 — SCATTER + PHOTO PAIRS
# =============================================================================

def make_figure_5(master: pd.DataFrame, selection: pd.DataFrame):
    dino = load_square(
        DINO_DIR / "04_prototype_similarity_reference.csv"
    ).to_numpy(float)

    _, ad_profiles = build_profiles(master, AD_SYMBOLS)
    ad_sim = cosine_similarity(ad_profiles)

    rho, pval = permutation_spearman(
        dino,
        ad_sim,
        permutations=PERMUTATIONS,
        seed=SEED,
    )

    rows = []
    for i, j in combinations(range(12), 2):
        sid1, sid2 = STYLE_IDS[i], STYLE_IDS[j]
        rows.append({
            "pair": f"{sid1}-{sid2}",
            "sid1": sid1,
            "sid2": sid2,
            "dino": float(dino[i, j]),
            "symbol": float(ad_sim[i, j]),
        })

    pairs = pd.DataFrame(rows)

    fig = plt.figure(figsize=(16.4, 10.6))
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.42, 1.0],
        wspace=0.13,
    )

    # -----------------------------------------------------------------
    # (a) scatter
    # -----------------------------------------------------------------
    ax = fig.add_subplot(outer[0, 0])

    ax.scatter(
        pairs["dino"],
        pairs["symbol"],
        s=44,
        color=REFERENCE,
        alpha=0.43,
        edgecolor="none",
    )

    x_med = pairs["dino"].median()
    y_med = pairs["symbol"].median()

    ax.axvline(
        x_med,
        linestyle="--",
        color=LIGHT_GREY,
        linewidth=1.0,
    )
    ax.axhline(
        y_med,
        linestyle="--",
        color=LIGHT_GREY,
        linewidth=1.0,
    )

    highlight_colors = {
        "01-04": HISTORICAL,
        "03-05": SYMBOL,
        "11-12": FIELD,
        "03-10": DINO,
    }

    texts = []
    for pair, color in highlight_colors.items():
        r = pairs[pairs["pair"].eq(pair)].iloc[0]

        ax.scatter(
            r["dino"],
            r["symbol"],
            s=92,
            color=color,
            edgecolor=BG,
            linewidth=0.9,
            zorder=4,
        )

        texts.append(
            ax.text(
                r["dino"],
                r["symbol"],
                pair,
                fontsize=9.0,
                color=TEXT,
                zorder=5,
            )
        )

    if HAS_ADJUSTTEXT:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="-",
                color=LIGHT_GREY,
                lw=0.7,
            ),
        )

    ax.text(
        0.03, 0.97,
        "Shared vocabulary\n+ divergent visual syntax",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.4,
        color=DARK_GREY,
    )
    ax.text(
        0.97, 0.97,
        "Shared vocabulary\n+ shared visual syntax",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.4,
        color=DARK_GREY,
    )
    ax.text(
        0.03, 0.03,
        "Distinct vocabulary\n+ distinct visual syntax",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.4,
        color=DARK_GREY,
    )
    ax.text(
        0.97, 0.03,
        "Similar visual syntax\n+ different component repertoire",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        color=DARK_GREY,
    )

    ax.text(
        0.975, 0.14,
        f"Spearman ρ = {rho:.3f}\n"
        f"Permutation p = {pval:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.4,
        color=TEXT,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=BG,
            edgecolor=GRID,
            linewidth=0.8,
        ),
    )

    ax.set_xlabel(
        "Whole-image visual syntax similarity\n(DINOv2 reference prototypes)",
        fontsize=11.5,
    )
    ax.set_ylabel(
        "Architectural vocabulary similarity\n(A–D symbol composition)",
        fontsize=11.5,
    )
    ax.set_xlim(
        max(0, pairs["dino"].min() - 0.05),
        min(1.0, pairs["dino"].max() + 0.05),
    )
    ax.set_ylim(
        max(0, pairs["symbol"].min() - 0.05),
        min(1.02, pairs["symbol"].max() + 0.03),
    )
    ax.grid(color=GRID, linewidth=0.7)
    clean_spines(ax)
    panel_label(ax, "(a)")

    # -----------------------------------------------------------------
    # (b) four photographic pair rows, each with its OWN descriptor axis
    # -----------------------------------------------------------------
    right = outer[0, 1].subgridspec(
        4, 1,
        hspace=0.22,
    )

    descriptors = {
        "01-04": "Concordant affinity",
        "03-05": "Shared vocabulary, divergent syntax",
        "11-12": "Strong two-layer affinity",
        "03-10": "Distinct vocabulary and syntax",
    }

    for rr, (sid1, sid2) in enumerate(FIG5_PAIRS):
        pair = f"{sid1}-{sid2}"
        pr = pairs[pairs["pair"].eq(pair)].iloc[0]

        # Separate text strip from image strip = no overlap.
        sub = right[rr, 0].subgridspec(
            2, 2,
            height_ratios=[0.18, 1.0],
            hspace=0.03,
            wspace=0.045,
        )

        desc_ax = fig.add_subplot(sub[0, :])
        desc_ax.axis("off")
        desc_ax.text(
            0.5, 0.50,
            f"{descriptors[pair]}   |   "
            f"DINO={pr['dino']:.3f}, A–D={pr['symbol']:.3f}",
            ha="center",
            va="center",
            fontsize=8.8,
            color=TEXT,
        )

        if rr == 0:
            panel_label(desc_ax, "(b)")

        for cc, sid in enumerate([sid1, sid2]):
            axp = fig.add_subplot(sub[1, cc])

            exemplar = get_exemplar_record(
                selection,
                sid,
            )
            path = str(exemplar["image_path"])
            exemplar_domain = str(exemplar["domain"])

            show_photo(
                axp,
                path,
                title=None,
                border_color=domain_color(exemplar_domain),
            )

            # Style name inside top-left; domain inside bottom-right.
            axp.text(
                0.02, 0.96,
                f"{sid}  {STYLE_NAMES[sid]}",
                transform=axp.transAxes,
                ha="left",
                va="top",
                fontsize=7.6,
                color=TEXT,
                bbox=dict(
                    boxstyle="round,pad=0.18",
                    facecolor=BG,
                    edgecolor="none",
                    alpha=0.88,
                ),
            )

            axp.text(
                0.97, 0.05,
                exemplar_domain.capitalize(),
                transform=axp.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.9,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.16",
                    facecolor=domain_color(exemplar_domain),
                    edgecolor="none",
                    alpha=0.90,
                ),
            )

    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.08,
        top=0.965,
    )

    save_figure(fig, "Figure_5_v4")


# =============================================================================
# 7. FIGURE 6 V2 — NETWORK + MULTI-LINEAGE PHOTO CASES
# =============================================================================

FIXED_POS = {
    "01": (-0.75, 0.40),
    "02": (-0.45, 0.00),
    "03": (-0.10, -0.42),
    "04": (-0.10, 0.42),
    "05": (0.38, 0.12),
    "06": (0.72, -0.30),
    "07": (0.55, 0.58),
    "08": (0.86, 0.78),
    "09": (0.82, 0.03),
    "10": (0.00, 0.90),
    "11": (0.28, -0.82),
    "12": (-0.38, -0.82),
}


def make_figure_6(master: pd.DataFrame, selection: pd.DataFrame):
    hist = load_square(
        STAGE4_DIR / "07_lineage_cooccurrence_jaccard_historical.csv"
    ).to_numpy(float)

    field = load_square(
        STAGE4_DIR / "07_lineage_cooccurrence_jaccard_field.csv"
    ).to_numpy(float)

    Y = y_matrix(master)

    domain_counts = {}
    for domain in ["historical", "field"]:
        m = master["domain"].eq(domain).to_numpy()
        domain_counts[domain] = Y[m].sum(axis=0)

    fig = plt.figure(figsize=(15.8, 10.4))
    outer = fig.add_gridspec(
        2, 2,
        height_ratios=[1.15, 0.85],
        hspace=0.16,
        wspace=0.10,
    )

    # network panels
    for col, (domain, M, node_color) in enumerate([
        ("historical", hist, HISTORICAL),
        ("field", field, FIELD),
    ]):
        ax = fig.add_subplot(outer[0, col])

        G = nx.Graph()
        G.add_nodes_from(STYLE_IDS)

        for i, j in combinations(range(12), 2):
            w = float(M[i, j])
            if w >= NETWORK_EDGE_THRESHOLD:
                G.add_edge(
                    STYLE_IDS[i],
                    STYLE_IDS[j],
                    weight=w,
                )

        counts = domain_counts[domain]
        sizes = 600 + 1500 * counts / max(counts.max(), 1)

        nx.draw_networkx_nodes(
            G,
            FIXED_POS,
            node_size=sizes,
            node_color=node_color,
            edgecolors=BG,
            linewidths=1.0,
            alpha=0.92,
            ax=ax,
        )

        widths = [
            0.9 + 8.0 * G[u][v]["weight"]
            for u, v in G.edges()
        ]

        nx.draw_networkx_edges(
            G,
            FIXED_POS,
            width=widths,
            edge_color=NEUTRAL,
            alpha=0.70,
            ax=ax,
        )

        nx.draw_networkx_labels(
            G,
            FIXED_POS,
            labels={sid: sid for sid in STYLE_IDS},
            font_family="Times New Roman",
            font_size=8,
            font_color="white",
            ax=ax,
        )

        ax.set_xlim(-1.02, 1.05)
        ax.set_ylim(-1.02, 1.05)
        ax.axis("off")
        ax.set_title(
            f"{domain.capitalize()} domain\nObserved lineage overlap (Jaccard ≥ {NETWORK_EDGE_THRESHOLD:.2f})",
            fontsize=11,
        )
        panel_label(ax, "(a)" if col == 0 else "(b)")

    # Compact lineage key between networks and photos.
    key_ax = fig.add_axes([0.21, 0.455, 0.58, 0.055])
    key_ax.axis("off")
    key_text_1 = "   ·   ".join(
        f"{sid} {STYLE_NAMES[sid]}"
        for sid in STYLE_IDS[:6]
    )
    key_text_2 = "   ·   ".join(
        f"{sid} {STYLE_NAMES[sid]}"
        for sid in STYLE_IDS[6:]
    )
    key_ax.text(
        0.5, 0.67,
        key_text_1,
        ha="center",
        va="center",
        fontsize=8.0,
        color=DARK_GREY,
    )
    key_ax.text(
        0.5, 0.28,
        key_text_2,
        ha="center",
        va="center",
        fontsize=8.0,
        color=DARK_GREY,
    )

    # Use four fixed, visually reviewed main-text cases.
    photo_cases = []

    for domain, pair in FIG6_CASES:
        rec = get_pair_record(
            selection,
            domain,
            pair,
        )
        photo_cases.append(
            (
                domain,
                pair,
                str(rec["image_path"]),
            )
        )


    bottom = outer[1, :].subgridspec(
        1, 4,
        wspace=0.08,
    )

    for i, (domain, pair, path) in enumerate(photo_cases[:4]):
        ax = fig.add_subplot(bottom[0, i])

        color = HISTORICAL if domain == "historical" else FIELD

        a, b = pair.split("-")

        show_photo(
            ax,
            path,
            title=f"{domain.capitalize()} | {a} + {b}",
            subtitle=f"{STYLE_NAMES[a]} + {STYLE_NAMES[b]}",
            border_color=color,
        )

        if i == 0:
            panel_label(ax, "(c)")

    fig.subplots_adjust(
        left=0.04,
        right=0.985,
        bottom=0.07,
        top=0.96,
    )

    save_figure(fig, "Figure_6_v4")


# =============================================================================
# 8. FIGURE 7 V2 — CONTINUITY + PHOTO TRIPLETS
# =============================================================================

def compute_ad_continuity(master: pd.DataFrame):
    Y = y_matrix(master)

    X = np.vstack(
        master["symbol_counts_json"]
        .map(lambda x: symbol_vector(x, AD_SYMBOLS))
        .to_numpy()
    )

    ref = master["domain"].eq("reference").to_numpy()
    single = Y.sum(axis=1) == 1
    canonical = ref & single

    ref_counts = np.zeros((12, len(AD_SYMBOLS)))

    for k in range(12):
        m = canonical & (Y[:, k] == 1)
        ref_counts[k] = X[m].sum(axis=0)

    ref_profiles = (
        ref_counts + ALPHA
    ) / (
        ref_counts.sum(axis=1, keepdims=True)
        + ALPHA * len(AD_SYMBOLS)
    )

    rows = []

    for domain in ["historical", "field"]:
        dmask = master["domain"].eq(domain).to_numpy()

        for k, sid in enumerate(STYLE_IDS):
            m = dmask & (Y[:, k] == 1)
            n = int(m.sum())

            counts = X[m].sum(axis=0)

            p = (
                counts + ALPHA
            ) / (
                counts.sum()
                + ALPHA * len(AD_SYMBOLS)
            )

            js = float(
                jensenshannon(
                    ref_profiles[k],
                    p,
                    base=2.0,
                )
            )

            rows.append({
                "style_id": sid,
                "domain": domain,
                "n_positive": n,
                "adequate_support": int(n >= MIN_SUPPORT),
                "AD_symbol_continuity": 1.0 - js,
            })

    return pd.DataFrame(rows)


def draw_dumbbell(
    ax,
    data: pd.DataFrame,
    value_col: str,
    title: str,
    xlim: Tuple[float, float],
    xlabel: str,
):
    hist = (
        data[data["domain"].eq("historical")]
        .set_index("style_id")
        .reindex(STYLE_IDS)
    )
    fld = (
        data[data["domain"].eq("field")]
        .set_index("style_id")
        .reindex(STYLE_IDS)
    )

    y = np.arange(12)

    for i, sid in enumerate(STYLE_IDS):
        hv = float(hist.loc[sid, value_col])
        fv = float(fld.loc[sid, value_col])

        ax.plot(
            [hv, fv],
            [i, i],
            color=LIGHT_GREY,
            linewidth=1.2,
            zorder=1,
        )

        h_ok = int(hist.loc[sid, "adequate_support"]) == 1
        f_ok = int(fld.loc[sid, "adequate_support"]) == 1

        ax.scatter(
            hv, i,
            marker="o",
            s=52,
            facecolor=HISTORICAL if h_ok else BG,
            edgecolor=HISTORICAL,
            linewidth=1.2,
            zorder=3,
        )

        ax.scatter(
            fv, i,
            marker="^",
            s=62,
            facecolor=FIELD if f_ok else BG,
            edgecolor=FIELD,
            linewidth=1.2,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([STYLE_TICK[s] for s in STYLE_IDS])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    clean_spines(ax)


def make_figure_7(master: pd.DataFrame, selection: pd.DataFrame):
    dino = read_csv(
        DINO_DIR / "05_domain_centroid_alignment.csv",
        dtype={"style_id": str},
    )

    dino["style_id"] = dino["style_id"].map(style_id_clean)
    dino["domain"] = dino["target_domain"]

    dino["adequate_support"] = (
        pd.to_numeric(
            dino["n_target_positive"],
            errors="coerce",
        ).fillna(0) >= MIN_SUPPORT
    ).astype(int)

    ad = compute_ad_continuity(master)

    fig = plt.figure(figsize=(16.4, 9.6))
    outer = fig.add_gridspec(
        2, 2,
        height_ratios=[1.0, 0.62],
        hspace=0.16,
        wspace=0.22,
    )

    ax1 = fig.add_subplot(outer[0, 0])
    draw_dumbbell(
        ax1,
        dino,
        value_col="centroid_cosine_similarity",
        title="Global visual-syntax alignment",
        xlim=(0, 0.72),
        xlabel="Reference–target centroid cosine similarity",
    )
    panel_label(ax1, "(a)")

    ax2 = fig.add_subplot(outer[0, 1])
    draw_dumbbell(
        ax2,
        ad,
        value_col="AD_symbol_continuity",
        title="Component-vocabulary continuity",
        xlim=(0.20, 0.70),
        xlabel="1 − Jensen–Shannon distance",
    )
    panel_label(ax2, "(b)")

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=HISTORICAL,
            markeredgecolor=HISTORICAL,
            markersize=7,
            label="Historical",
        ),
        Line2D(
            [0], [0],
            marker="^",
            color="none",
            markerfacecolor=FIELD,
            markeredgecolor=FIELD,
            markersize=8,
            label="Field",
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=BG,
            markeredgecolor=DARK_GREY,
            markersize=7,
            label=f"Low support (N < {MIN_SUPPORT})",
        ),
    ]

    # Put legend in the inter-row gap, centered and compact.
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.445),
        fontsize=9.5,
    )

    # -----------------------------------------------------------------
    # Photo triplets, each with a dedicated group-title strip.
    # -----------------------------------------------------------------
    bottom = outer[1, :].subgridspec(
        1, 3,
        wspace=0.085,
    )

    for ci, sid in enumerate(FIG7_LINEAGES):
        group = bottom[0, ci].subgridspec(
            2, 3,
            height_ratios=[0.15, 1.0],
            hspace=0.03,
            wspace=0.035,
        )

        title_ax = fig.add_subplot(group[0, :])
        title_ax.axis("off")
        title_ax.text(
            0.5, 0.5,
            f"{sid}  {STYLE_NAMES[sid]}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=TEXT,
        )

        if ci == 0:
            panel_label(title_ax, "(c)")

        for dj, (domain, color) in enumerate([
            ("reference", REFERENCE),
            ("historical", HISTORICAL),
            ("field", FIELD),
        ]):
            axp = fig.add_subplot(group[1, dj])

            path = get_lineage_photo(
                selection,
                domain,
                sid,
            )

            show_photo(
                axp,
                path,
                title=domain.capitalize(),
                subtitle=None,
                border_color=color,
            )

    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.060,
        top=0.955,
    )

    save_figure(fig, "Figure_7_v4")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=None,
        help="Optional full path to 03_manual_selection.csv",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("PAPER 1 — FINAL FIGURE 3–7 V4")
    print("=" * 100)

    # Core files check.
    required = [
        CONVNEXT_DIR / "cv_fold_metrics.csv",
        CONVNEXT_DIR / "final_metrics.json",
        CONVNEXT_DIR / "historical_external_per_label_metrics.csv",
        CONVNEXT_DIR / "field_external_per_label_metrics.csv",
        DINO_DIR / "04_prototype_similarity_reference.csv",
        DINO_DIR / "05_domain_centroid_alignment.csv",
        DINO_DIR / "06_prototype_external_metrics.csv",
        DINO_DIR / "07_linear_probe_external_metrics.csv",
        DINO_DIR / "08_per_lineage_external_metrics.csv",
        STAGE4_DIR / "07_lineage_cooccurrence_jaccard_historical.csv",
        STAGE4_DIR / "07_lineage_cooccurrence_jaccard_field.csv",
        FROZEN_DIR / "01_unique_image_multilabel_master.csv",
    ]

    for p in required:
        require(p)

    selection = load_selection(args.selection_file)
    master = load_master()

    make_figure_3()
    make_figure_4(master, selection)
    make_supplementary_railway_context(master)
    make_figure_5(master, selection)
    make_figure_6(master, selection)
    make_figure_7(master, selection)

    summary = """
FINAL V4 FIGURES COMPLETE

Figure 3  Quantitative model performance and external transfer
Figure 4  Reviewed architectural exemplar atlas + separated A–D signature matrix
Figure 5  Two-layer visual genealogy + non-overlapping photographic pair cases
Figure 6  Historical/field overlap networks + fixed reviewed multi-lineage cases
Figure 7  Cross-domain continuity + compact reference/historical/field triplets
Figure S4 Railway-infrastructural context (E1–E15)

No figure captions are embedded inside the figures.\nFigure 4/5 use visually reviewed illustrative exemplars; quantitative canonical profiles remain reference-based.
"""
    (OUT_DIR / "00_v2_figure_summary.txt").write_text(
        summary.strip() + "\n",
        encoding="utf-8",
    )

    print(summary)
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
