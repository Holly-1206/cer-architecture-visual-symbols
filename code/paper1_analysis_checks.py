# -*- coding: utf-8 -*-
r"""
paper1_analysis_checks.py

Paper 1 — robustness checks requested in the advisor review (DY)
=================================================================

Four checks, all run on the FINAL frozen v3 dataset and the saved model
outputs. Nothing is retrained and no original file is modified.

CHECK 1  leakage     (comments C4, C8)
    Exact-pixel deduplication cannot detect the same building photographed
    from another angle, cropped differently, or rescanned. This check builds
    image groups from
      (a) visual near-duplicates: frozen DINOv2 cosine >= --dino-threshold,
          or dHash Hamming <= --dhash-threshold with cosine >= 0.70;
      (b) building / site names parsed from album and field filenames;
      (c) camera-number sequences for IMG_xxxx field photographs.
    It then reports
      - near-duplicate groups that cross reference / historical / field;
      - reference validation images whose group also has a TRAINING image
        in another fold, and CV metrics recomputed without them;
      - external images whose group contains a reference image, and
        external metrics recomputed without them;
      - a group-aware 5-fold split, written as a drop-in dataset folder
        for an optional ConvNeXt re-run with train_paper1_multilabel.py.

CHECK 2  multilabel  (comment C6)
    Is a multi-label image ONE building carrying several lineages, or
    several buildings / objects in one frame? Each lineage has its own
    LabelMe JSON, so the spatial footprint of each lineage's shapes can be
    compared. The heuristic classes are checked by a stratified manual
    coding sample (contact sheets + CSV template). The observed
    co-occurrence analysis is re-run without likely multi-object images.

CHECK 3  density     (comment C11)
    Re-runs the A–D vocabulary layer with three counting units:
      instance          symbol instances (published analysis)
      presence          each symbol counted once per image
      image_normalised  each image contributes total weight 1
    and reports the effect on the A–D matrix, the DINOv2–A–D correlation,
    the four relational regions, key pairs and A–D continuity. Adds an
    equal-n subsampling test and a joint bootstrap CI for the two-layer
    correlation (comment C14).

CHECK 4  bootstrap   (comment C13)
    Paired bootstrap of external-domain macro-AUROC and mAP for ConvNeXt,
    DINOv2 prototypes and the DINOv2 linear probe, both image-level and
    cluster-level (resampling building / site groups from CHECK 1).

RUN AFTER
---------
repair_paper1_freeze_v3.py, train_paper1_multilabel.py,
paper1_dinov2_crossdomain.py, paper1_visual_genealogy_final.py

Recommended command
-------------------
python paper1_analysis_checks.py

Only some checks:
python paper1_analysis_checks.py --checks density bootstrap

Optional manual coding for CHECK 2:
    open 12_manual_coding_sample.csv, fill `manual_unit` using the contact
    sheets, save it as 12_manual_coding_sample_FILLED.csv in the same
    folder, and run `--checks multilabel` again.

Dependencies
------------
pip install numpy pandas scipy scikit-learn pillow
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 0. CONFIG
# =============================================================================

BASE_DIR = Path(str(PROJECT_ROOT))
DEFAULT_DATA_DIR = BASE_DIR / "paper1_frozen_dataset_v3"
DEFAULT_MODEL_DIR = BASE_DIR / "paper1_model_results" / "convnext_tiny"
DEFAULT_DINO_DIR = BASE_DIR / "paper1_dinov2_results"
DEFAULT_GENEALOGY_DIR = BASE_DIR / "paper1_visual_genealogy_results"
DEFAULT_OUT_DIR = Path(
    str(PROJECT_ROOT / "results" / "paper1_analysis_checks_results")
)

STYLE_IDS = [f"{i:02d}" for i in range(1, 13)]

STYLE_NAMES_EN = {
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

AD_SYMBOLS = (
    [f"A{i}" for i in range(1, 13)]
    + [f"B{i}" for i in range(1, 17)]
    + [f"C{i}" for i in range(1, 21)]
    + [f"D{i}" for i in range(1, 22)]
)
E_SYMBOLS = [f"E{i}" for i in range(1, 16)]
ALL_KNOWN_SYMBOLS = set(AD_SYMBOLS + E_SYMBOLS)

ROLE_TO_DOMAIN = {
    "development_reference": "reference",
    "external_historical": "historical",
    "external_field": "field",
}
DOMAINS = ["reference", "historical", "field"]

EMBEDDING_FILES = {
    "reference": ("01_reference_embedding_meta.csv", "embeddings_reference.npy"),
    "historical": ("02_historical_embedding_meta.csv", "embeddings_historical.npy"),
    "field": ("03_field_embedding_meta.csv", "embeddings_field.npy"),
}

DIRICHLET_ALPHA = 0.5
MIN_DOMAIN_LINEAGE_N = 15
SEED = 20261010

PAIRS = list(combinations(range(12), 2))

# Pairs discussed in the manuscript (Table 7 and Section 4.4).
KEY_PAIRS = [
    ("01", "04"),
    ("11", "12"),
    ("06", "11"),
    ("03", "05"),
    ("04", "07"),
    ("05", "09"),
    ("03", "10"),
]

REGION_LABELS = [
    "High–High",
    "High vocabulary / Low syntax",
    "Low vocabulary / High syntax",
    "Low–Low",
]

# View words removed so that "X正面" and "X全景" fall into the same building.
VIEW_WORDS = [
    "正面", "侧面", "背面", "全景", "近景", "远景", "远眺", "细部",
    "局部", "内部", "外观", "立面", "俯瞰", "鸟瞰", "入口",
]

# Filename words indicating a scene / street / group view rather than a building.
SCENE_PATTERN = re.compile(
    r"全景|全貌|群|街|拼版|图版|混合|一带|站区|站场|市区|鸟瞰|远眺|俯瞰|聚落|院落"
)

MANUAL_CODES = [
    "single_building",
    "multiple_buildings",
    "building_plus_context",
    "unclear",
]

LINEAGE_COLOURS = [
    (230, 57, 70),
    (29, 120, 220),
    (46, 160, 67),
    (245, 166, 35),
    (150, 80, 200),
]


# =============================================================================
# 1. BASIC HELPERS
# =============================================================================

@dataclass
class Context:
    args: argparse.Namespace
    df: pd.DataFrame
    Y: np.ndarray
    X_ad: np.ndarray
    emb: np.ndarray
    out_dir: Path
    report: Dict[str, List[str]] = field(default_factory=dict)
    groups: pd.DataFrame | None = None


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def parse_counts(value) -> Dict[str, int]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        obj = json.loads(value)
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}

    out = {}
    for key, val in obj.items():
        key = str(key).strip()
        if key not in ALL_KNOWN_SYMBOLS:
            continue
        try:
            n = int(float(val))
        except Exception:
            continue
        if n > 0:
            out[key] = n
    return out


def upper_tri_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = spearmanr(x, y)
    return float(getattr(result, "statistic", result[0]))


def matrix_permutation_spearman(
    A: np.ndarray,
    B: np.ndarray,
    n_perm: int,
    seed: int,
) -> Tuple[float, float]:
    """Same matrix-concordance test as paper1_visual_genealogy_final.py."""
    a = upper_tri_values(A)
    observed = safe_spearman(a, upper_tri_values(B))

    rng = np.random.default_rng(seed)
    hits = 0
    n = B.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n)
        rp = safe_spearman(a, upper_tri_values(B[np.ix_(perm, perm)]))
        if abs(rp) >= abs(observed):
            hits += 1

    return observed, float((hits + 1) / (n_perm + 1))


def smoothed_profile(counts: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    if counts.ndim == 1:
        return (counts + alpha) / (counts.sum() + alpha * counts.shape[0])
    return (counts + alpha) / (
        counts.sum(axis=1, keepdims=True) + alpha * counts.shape[1]
    )


def load_square_matrix(path: Path) -> np.ndarray:
    """Load a 12x12 lineage matrix and reorder rows/columns to 01...12."""
    frame = pd.read_csv(require(path), index_col=0, encoding="utf-8-sig")

    def sid(label) -> str | None:
        m = re.match(r"^\s*(0[1-9]|1[0-2])(?!\d)", str(label))
        return m.group(1) if m else None

    rows = {sid(i): i for i in frame.index}
    cols = {sid(c): c for c in frame.columns}
    if all(s in rows for s in STYLE_IDS) and all(s in cols for s in STYLE_IDS):
        frame = frame.loc[[rows[s] for s in STYLE_IDS], [cols[s] for s in STYLE_IDS]]

    matrix = frame.to_numpy(dtype=float)
    if matrix.shape != (12, 12):
        raise RuntimeError(f"Expected a 12x12 matrix: {path}")
    return matrix


def classify_regions(dino: np.ndarray, ad: np.ndarray) -> List[str]:
    """Median-split relational regions, as in the published Table S3."""
    d = upper_tri_values(dino)
    a = upper_tri_values(ad)
    d_med, a_med = np.median(d), np.median(a)

    labels = []
    for dv, av in zip(d, a):
        syntax_high = dv >= d_med
        vocab_high = av >= a_med
        if syntax_high and vocab_high:
            labels.append("High–High")
        elif vocab_high:
            labels.append("High vocabulary / Low syntax")
        elif syntax_high:
            labels.append("Low vocabulary / High syntax")
        else:
            labels.append("Low–Low")
    return labels


def pair_index(sid_a: str, sid_b: str) -> int:
    i, j = sorted((STYLE_IDS.index(sid_a), STYLE_IDS.index(sid_b)))
    return PAIRS.index((i, j))


# -----------------------------------------------------------------------------
# Ranking metrics (tie-aware, identical to sklearn, but fast enough for
# thousands of bootstrap replicates).
# -----------------------------------------------------------------------------

def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_ap(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(y.sum())
    if n1 == 0:
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    ends = np.r_[np.where(np.diff(s_sorted))[0], len(y_sorted) - 1]
    tps = np.cumsum(y_sorted)[ends]
    fps = (ends + 1) - tps
    precision = tps / (tps + fps)
    recall = tps / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def per_class_scores(Y: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    aucs = np.array([fast_auc(Y[:, k], S[:, k]) for k in range(Y.shape[1])])
    aps = np.array([fast_ap(Y[:, k], S[:, k]) for k in range(Y.shape[1])])
    return aucs, aps


def ranking_metrics(Y: np.ndarray, S: np.ndarray) -> Dict[str, float]:
    aucs, aps = per_class_scores(Y, S)
    return {
        "macro_AUROC": float(np.nanmean(aucs)),
        "mAP": float(np.nanmean(aps)),
    }


def threshold_metrics(Y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(Y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
    }


# -----------------------------------------------------------------------------
# Contact sheets
# -----------------------------------------------------------------------------

def get_font(size: int = 14):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def open_rgb(path: str) -> Image.Image:
    try:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        return Image.new("RGB", (300, 300), (200, 200, 200))


def save_contact_sheet(
    tiles: List[Image.Image],
    captions: List[str],
    path: Path,
    ncols: int,
    tile_size: Tuple[int, int],
) -> None:
    if not tiles:
        return
    font = get_font(14)
    caption_h = 40
    w, h = tile_size
    nrows = int(np.ceil(len(tiles) / ncols))
    sheet = Image.new("RGB", (ncols * w, nrows * (h + caption_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for k, (tile, caption) in enumerate(zip(tiles, captions)):
        r, c = divmod(k, ncols)
        thumb = ImageOps.contain(tile, (w - 8, h - 8))
        x0 = c * w + (w - thumb.width) // 2
        y0 = r * (h + caption_h) + (h - thumb.height) // 2
        sheet.paste(thumb, (x0, y0))
        draw.multiline_text(
            (c * w + 6, r * (h + caption_h) + h + 2),
            caption,
            fill="black",
            font=font,
        )

    sheet.save(path, quality=90)


def side_by_side(a: Image.Image, b: Image.Image, height: int = 300) -> Image.Image:
    a = ImageOps.contain(a, (10_000, height))
    b = ImageOps.contain(b, (10_000, height))
    out = Image.new("RGB", (a.width + b.width + 6, height), "white")
    out.paste(a, (0, 0))
    out.paste(b, (a.width + 6, 0))
    return out


# =============================================================================
# 2. LOAD FROZEN DATA + EMBEDDINGS
# =============================================================================

def load_context(args) -> Context:
    master_path = require(args.data_dir / "01_unique_image_multilabel_master.csv")
    master = pd.read_csv(master_path, encoding="utf-8-sig")

    df = (
        master[master["dataset_role"].isin(ROLE_TO_DOMAIN)]
        .copy()
        .reset_index(drop=True)
    )
    df["domain"] = df["dataset_role"].map(ROLE_TO_DOMAIN)
    df["stem"] = df["annotation_master_json"].map(lambda p: Path(str(p)).stem)
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")

    Y = df[[f"y_{sid}" for sid in STYLE_IDS]].to_numpy(dtype=int)

    X_ad = np.vstack(
        df["symbol_counts_json"].map(
            lambda v: np.array(
                [parse_counts(v).get(sym, 0) for sym in AD_SYMBOLS],
                dtype=float,
            )
        ).to_numpy()
    )

    emb_map = {}
    for domain, (meta_name, npy_name) in EMBEDDING_FILES.items():
        meta = pd.read_csv(require(args.dino_dir / meta_name), encoding="utf-8-sig")
        arr = np.load(require(args.dino_dir / npy_name))
        for sha, row in zip(meta["pixel_sha256"], meta["embedding_row"]):
            emb_map[(domain, sha)] = arr[int(row)]

    missing = [
        (d, s)
        for d, s in zip(df["domain"], df["pixel_sha256"])
        if (d, s) not in emb_map
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} analytical images have no DINOv2 embedding.")

    emb = np.vstack(
        [emb_map[(d, s)] for d, s in zip(df["domain"], df["pixel_sha256"])]
    ).astype(np.float32)

    print("Frozen analytical images:")
    for domain in DOMAINS:
        print(f"  {domain:<10}: {int((df['domain'] == domain).sum()):,}")
    print(f"  total     : {len(df):,}\n")

    return Context(
        args=args,
        df=df,
        Y=Y,
        X_ad=X_ad,
        emb=emb,
        out_dir=args.out_dir,
    )


# =============================================================================
# 3. CHECK 1 — BUILDING / SITE GROUPS AND LEAKAGE
# =============================================================================

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def normalise_subject(subject: str) -> str:
    s = subject.strip()
    previous = None
    while previous != s:
        previous = s
        s = re.sub(r"[（(]\s*\d+\s*[)）]$", "", s)
        s = re.sub(r"[\s\d一二三四五六七八九十]+$", "", s)
        for word in VIEW_WORDS:
            if s.endswith(word) and len(s) > len(word) + 1:
                s = s[: -len(word)]
    return s or subject.strip()


def historical_building_key(stem: str) -> str:
    """
    Album filenames:  D153_横道河子_铁路线路与站场_一面坡站站台与站区(1)
                      code_place_category_subject[__lineage notes]
    Key = place | subject without numbering and view words.
    """
    s = re.sub(r"\s*-\s*副本$", "", stem)
    s = re.sub(r"^[A-Za-z]?\d+_", "", s)
    s = s.split("__")[0]
    parts = [p for p in s.split("_") if p]

    if len(parts) >= 3:
        place, subject = parts[0], parts[2]
    elif len(parts) == 2:
        place, subject = parts[0], parts[1]
    else:
        place, subject = "", (parts[0] if parts else stem)

    place = re.sub(r"[（(].*?[)）]", "", place)
    return f"{place}|{normalise_subject(subject)}"


def field_site_key(stem: str) -> Tuple[str | None, str, int | None]:
    """
    Returns (site_key, provenance, camera_number).
    Named field photos (安达7, 新城子站1) are keyed by station / site.
    IMG_xxxx photos are keyed later by camera-number sequence.
    """
    if re.match(r"^(截屏|屏幕截图|Screenshot)", stem, re.I):
        return f"screenshot|{stem}", "screenshot", None
    if "~tplv" in stem or re.match(r"^[0-9a-f]{24,}", stem):
        return f"web|{stem}", "web_image", None

    m = re.match(r"^IMG_(\d+)", stem, re.I)
    if m:
        return None, "camera_sequence", int(m.group(1))

    s = re.sub(r"[（(].*?[)）]", "", stem)
    s = re.sub(r"[\s_\-]*\d+$", "", s).strip()
    if "站" in s and s.split("站")[0]:
        s = s.split("站")[0]
    return f"site|{s}", "named_site", None


def dhash_bits(series: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    bits = np.zeros((len(series), 64), dtype=np.float32)
    valid = np.zeros(len(series), dtype=bool)
    for i, value in enumerate(series):
        try:
            v = int(str(value).strip(), 16)
        except Exception:
            continue
        raw = np.frombuffer(v.to_bytes(8, "big"), dtype=np.uint8)
        bits[i] = np.unpackbits(raw).astype(np.float32)
        valid[i] = True
    return bits, valid


def build_groups(ctx: Context) -> pd.DataFrame:
    if ctx.groups is not None:
        return ctx.groups

    args = ctx.args
    df = ctx.df
    n = len(df)
    uf = UnionFind(n)

    # ---- (b)+(c) names / sites / camera sequences --------------------------
    name_key = [""] * n
    provenance = [""] * n
    camera = {}

    for i, row in df.iterrows():
        if row["domain"] == "historical":
            name_key[i] = "hist|" + historical_building_key(row["stem"])
            provenance[i] = "album_page"
        elif row["domain"] == "field":
            key, prov, number = field_site_key(row["stem"])
            provenance[i] = prov
            if key is None:
                camera[i] = number
            else:
                name_key[i] = "field|" + key
        else:
            name_key[i] = f"ref|{row['stem']}"
            provenance[i] = "reference_web" if "openverse" in row["stem"] else "reference_other"

    if camera:
        ordered = sorted(camera.items(), key=lambda kv: kv[1])
        start = prev = ordered[0][1]
        members = []
        for idx, number in ordered:
            if number - prev > args.img_seq_gap and members:
                for m in members:
                    name_key[m] = f"field|imgseq|{start}-{prev}"
                members, start = [], number
            members.append(idx)
            prev = number
        for m in members:
            name_key[m] = f"field|imgseq|{start}-{prev}"

    by_key: Dict[str, List[int]] = {}
    for i, key in enumerate(name_key):
        by_key.setdefault(key, []).append(i)
    for members in by_key.values():
        for m in members[1:]:
            uf.union(members[0], m)

    # ---- (a) visual near-duplicates -----------------------------------------
    print("[CHECK 1] Computing visual near-duplicate edges ...")
    S = ctx.emb @ ctx.emb.T
    bits, valid = dhash_bits(df["dhash"])
    H = bits @ (1.0 - bits).T + (1.0 - bits) @ bits.T
    H[~valid, :] = 99
    H[:, ~valid] = 99

    edges = []
    nn_count = np.zeros(n, dtype=int)
    for i in range(n - 1):
        cos = S[i, i + 1:]
        ham = H[i, i + 1:]
        hit = (cos >= args.dino_threshold) | ((ham <= args.dhash_threshold) & (cos >= 0.70))
        for off in np.nonzero(hit)[0]:
            j = i + 1 + int(off)
            uf.union(i, j)
            nn_count[i] += 1
            nn_count[j] += 1
            edges.append({
                "row_a": i,
                "row_b": j,
                "domain_a": df.at[i, "domain"],
                "domain_b": df.at[j, "domain"],
                "cross_domain": int(df.at[i, "domain"] != df.at[j, "domain"]),
                "dino_cosine": float(S[i, j]),
                "dhash_hamming": int(H[i, j]),
                "fold_a": df.at[i, "fold"],
                "fold_b": df.at[j, "fold"],
                "style_ids_a": df.at[i, "style_ids"],
                "style_ids_b": df.at[j, "style_ids"],
                "image_a": df.at[i, "image_path"],
                "image_b": df.at[j, "image_path"],
            })
    del S, H

    roots = [uf.find(i) for i in range(n)]
    root_to_gid = {r: g for g, r in enumerate(sorted(set(roots)))}
    group_id = np.array([root_to_gid[r] for r in roots])

    groups = pd.DataFrame({
        "row": np.arange(n),
        "domain": df["domain"],
        "pixel_sha256": df["pixel_sha256"],
        "stem": df["stem"],
        "fold": df["fold"],
        "style_ids": df["style_ids"],
        "name_key": name_key,
        "provenance": provenance,
        "n_visual_neighbours": nn_count,
        "group_id": group_id,
        "image_path": df["image_path"],
    })

    stats = groups.groupby("group_id").agg(
        group_size=("row", "size"),
        group_domains=("domain", lambda x: "|".join(sorted(set(x)))),
        group_n_reference=("domain", lambda x: int((x == "reference").sum())),
    )
    groups = groups.merge(stats, left_on="group_id", right_index=True, how="left")
    groups["group_n_domains"] = groups["group_domains"].str.count(r"\|") + 1

    ctx.groups = groups
    ctx.groups_edges = pd.DataFrame(edges)
    return groups


def assign_grouped_folds(
    units: List[Tuple[List[int], np.ndarray]],
    n_folds: int,
    seed: int,
) -> Dict[int, int]:
    """
    Group-aware version of assign_balanced_multilabel_folds()
    (repair_paper1_freeze_v3.py): whole groups are assigned to folds.
    """
    rng = np.random.default_rng(seed)
    totals = np.sum([u[1] for u in units], axis=0)
    targets = totals / n_folds
    n_images = sum(len(u[0]) for u in units)
    target_size = n_images / n_folds

    def order_key(unit):
        rarity = float(np.sum(unit[1] / np.maximum(totals, 1)))
        return (len(unit[0]), rarity, rng.random())

    ordered = sorted(units, key=order_key, reverse=True)
    fold_counts = np.zeros((n_folds, len(totals)))
    fold_sizes = np.zeros(n_folds)
    assignment = {}

    for members, labels in ordered:
        best_fold, best_score = 0, None
        for f in range(n_folds):
            deficit = np.maximum(targets - fold_counts[f], 0.0) / np.maximum(targets, 1e-12)
            label_need = float(np.sum(deficit * (labels > 0)))
            size_need = max(target_size - fold_sizes[f], 0.0) / max(target_size, 1.0)
            overflow = max(fold_sizes[f] + len(members) - np.ceil(target_size), 0)
            score = label_need + 0.5 * size_need - 10.0 * overflow + rng.random() * 1e-10
            if best_score is None or score > best_score:
                best_fold, best_score = f, score
        for m in members:
            assignment[m] = best_fold
        fold_sizes[best_fold] += len(members)
        fold_counts[best_fold] += labels

    return assignment


def check_leakage(ctx: Context) -> None:
    args = ctx.args
    out = ctx.out_dir / "check1_leakage"
    out.mkdir(parents=True, exist_ok=True)
    rep = ctx.report.setdefault("leakage", [])

    groups = build_groups(ctx)
    edges = ctx.groups_edges
    df = ctx.df
    Y = ctx.Y

    save_csv(groups.drop(columns=["image_path"]), out / "01_image_groups.csv")
    save_csv(edges, out / "02_near_duplicate_pairs.csv")

    # ---- group / site counts per domain --------------------------------------
    summary_rows = []
    for domain in DOMAINS:
        g = groups[groups["domain"] == domain]
        summary_rows.append({
            "domain": domain,
            "n_images": len(g),
            "n_name_or_site_keys": g["name_key"].nunique(),
            "n_groups_after_visual_merge": g["group_id"].nunique(),
            "n_images_with_visual_neighbour": int((g["n_visual_neighbours"] > 0).sum()),
            "n_images_in_cross_domain_groups": int((g["group_n_domains"] > 1).sum()),
        })
    domain_summary = pd.DataFrame(summary_rows)
    save_csv(domain_summary, out / "03_domain_group_summary.csv")

    # ---- field provenance ----------------------------------------------------
    field_g = groups[groups["domain"] == "field"]
    prov = (
        field_g.groupby("provenance")
        .agg(n_images=("row", "size"), style_ids=("style_ids", lambda x: "; ".join(sorted(set(x)))))
        .reset_index()
    )
    save_csv(prov, out / "04_field_provenance_summary.csv")
    non_photo = field_g[field_g["provenance"].isin(["screenshot", "web_image"])]
    save_csv(
        non_photo[["stem", "provenance", "style_ids", "image_path"]],
        out / "05_field_non_photograph_items.csv",
    )

    # ---- cross-domain near-duplicates ---------------------------------------
    cross = edges[edges["cross_domain"] == 1].sort_values("dino_cosine", ascending=False) if len(edges) else edges
    if len(cross):
        tiles, caps = [], []
        for k, (_, e) in enumerate(cross.head(args.max_sheet_pairs).iterrows(), 1):
            tiles.append(side_by_side(open_rgb(e["image_a"]), open_rgb(e["image_b"])))
            caps.append(
                f"#{k} {e['domain_a']} vs {e['domain_b']}\n"
                f"cos={e['dino_cosine']:.3f} dHash={e['dhash_hamming']}"
            )
        save_contact_sheet(tiles, caps, out / "06_cross_domain_pairs_sheet.jpg", 3, (620, 300))
        save_csv(
            cross.head(args.max_sheet_pairs).assign(sheet_no=lambda x: range(1, len(x) + 1)),
            out / "06_cross_domain_pairs_sheet_index.csv",
        )

    # ---- within-reference CV leakage ----------------------------------------
    ref_rows = groups[groups["domain"] == "reference"]
    fold_of = dict(zip(ref_rows["row"], ref_rows["fold"].astype(int)))
    members_by_group = ref_rows.groupby("group_id")["row"].apply(list).to_dict()

    leaked = {}
    for row, fold in fold_of.items():
        gid = groups.at[row, "group_id"]
        leaked[row] = any(fold_of[m] != fold for m in members_by_group.get(gid, []) if m != row)

    sha_to_row = {
        sha: r for sha, r in zip(ref_rows["pixel_sha256"], ref_rows["row"])
    }

    cv_rows = []
    for fold in range(5):
        pred_path = args.model_dir / f"fold_{fold}_validation_predictions.csv"
        thr_path = args.model_dir / f"fold_{fold}_per_label_metrics.csv"
        if not (pred_path.exists() and thr_path.exists()):
            print(f"  [WARN] missing fold {fold} predictions; CV leakage metrics skipped.")
            cv_rows = []
            break

        pred = pd.read_csv(pred_path, encoding="utf-8-sig")
        thr_df = pd.read_csv(thr_path, encoding="utf-8-sig", dtype={"style_id": str})
        thr = thr_df.set_index(thr_df["style_id"].str.zfill(2)).loc[STYLE_IDS, "threshold"].to_numpy(float)

        yv = pred[[f"y_{s}" for s in STYLE_IDS]].to_numpy(int)
        pv = pred[[f"p_{s}" for s in STYLE_IDS]].to_numpy(float)
        is_leaked = np.array([leaked[sha_to_row[s]] for s in pred["pixel_sha256"]])

        for subset, mask in [("all", np.ones(len(pred), bool)), ("clean", ~is_leaked)]:
            m = {
                **threshold_metrics(yv[mask], (pv[mask] >= thr).astype(int)),
                **ranking_metrics(yv[mask], pv[mask]),
            }
            cv_rows.append({
                "fold": fold,
                "subset": subset,
                "n_validation": int(mask.sum()),
                "n_leaked_removed": int(is_leaked.sum()) if subset == "clean" else 0,
                **m,
            })

    if cv_rows:
        cv = pd.DataFrame(cv_rows)
        save_csv(cv, out / "07_reference_cv_all_vs_clean.csv")
        cv_summary = (
            cv.groupby("subset")[["n_validation", "macro_f1", "micro_f1", "mAP", "macro_AUROC"]]
            .agg(["mean", "std"])
        )
        cv_summary.to_csv(out / "08_reference_cv_summary.csv", encoding="utf-8-sig")

    # ---- external images with a reference near-duplicate --------------------
    ext_rows = []
    P = prototypes_from_reference(ctx.emb[(df["domain"] == "reference").to_numpy()],
                                  Y[(df["domain"] == "reference").to_numpy()])
    for domain, pred_name in [
        ("historical", "historical_external_ensemble_predictions.csv"),
        ("field", "field_external_ensemble_predictions.csv"),
    ]:
        dmask = (df["domain"] == domain).to_numpy()
        rows_d = np.nonzero(dmask)[0]
        contaminated = groups.loc[rows_d, "group_n_reference"].to_numpy() > 0

        conv = load_convnext_external(args.model_dir / pred_name, df.loc[rows_d, "pixel_sha256"])
        Yd = Y[rows_d]
        Sd = ctx.emb[rows_d] @ P.T

        for subset, mask in [("all", np.ones(len(rows_d), bool)), ("clean", ~contaminated)]:
            row = {"domain": domain, "subset": subset, "n_images": int(mask.sum())}
            if conv is not None:
                cm = ranking_metrics(Yd[mask], conv[0][mask])
                tm = threshold_metrics(Yd[mask], conv[1][mask])
                row.update({f"ConvNeXt_{k}": v for k, v in {**tm, **cm}.items()})
            pm = ranking_metrics(Yd[mask], Sd[mask])
            row.update({f"DINOv2_prototype_{k}": v for k, v in pm.items()})
            ext_rows.append(row)

    ext = pd.DataFrame(ext_rows)
    save_csv(ext, out / "09_external_all_vs_clean.csv")

    # ---- album <-> field site matches (candidates for case readings) ---------
    hist = groups[groups["domain"] == "historical"]
    match_rows = []
    sites = field_g[field_g["provenance"] == "named_site"]["name_key"].str.replace("field|site|", "", regex=False)
    for site, n_field in sites.value_counts().items():
        if len(site) < 2:
            continue
        hits = hist[hist["stem"].str.contains(re.escape(site), regex=True)]
        if len(hits):
            match_rows.append({
                "field_site": site,
                "n_field_images": int(n_field),
                "n_album_pages": len(hits),
                "album_examples": " || ".join(hits["stem"].head(6)),
            })
    if match_rows:
        save_csv(
            pd.DataFrame(match_rows).sort_values(["n_album_pages", "n_field_images"], ascending=False),
            out / "10_candidate_album_field_site_matches.csv",
        )

    # ---- group-aware folds for an optional ConvNeXt re-run -------------------
    n_cross_fold_groups = 0
    if not args.no_grouped_dataset:
        ref_idx = ref_rows["row"].to_numpy()
        units = []
        for gid, members in ref_rows.groupby("group_id")["row"]:
            members = list(members)
            units.append((members, Y[members].sum(axis=0)))
            if len({fold_of[m] for m in members}) > 1:
                n_cross_fold_groups += 1

        assignment = assign_grouped_folds(units, 5, args.seed)
        grouped_dir = out / "paper1_frozen_dataset_v3_grouped"
        grouped_dir.mkdir(exist_ok=True)

        ref_csv = pd.read_csv(
            require(args.data_dir / "02_reference_development_5fold.csv"),
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
        )
        new_fold = {df.at[r, "pixel_sha256"]: assignment[r] for r in ref_idx}
        gid_of = {df.at[r, "pixel_sha256"]: int(groups.at[r, "group_id"]) for r in ref_idx}
        ref_csv["fold_original_v3"] = ref_csv["fold"]
        ref_csv["fold"] = ref_csv["pixel_sha256"].map(new_fold).astype(int).astype(str)
        ref_csv["leakage_group_id"] = ref_csv["pixel_sha256"].map(gid_of)
        save_csv(ref_csv, grouped_dir / "02_reference_development_5fold.csv")
        for name in ("03_historical_external.csv", "04_field_external.csv"):
            shutil.copy2(require(args.data_dir / name), grouped_dir / name)

        balance = []
        for f in range(5):
            members = [r for r in ref_idx if assignment[r] == f]
            balance.append({
                "fold": f,
                "n_images": len(members),
                **{f"n_{s}": int(Y[members, k].sum()) for k, s in enumerate(STYLE_IDS)},
            })
        save_csv(pd.DataFrame(balance), grouped_dir / "09_fold_balance_grouped.csv")

    # ---- report --------------------------------------------------------------
    n_ref = len(ref_rows)
    n_leak = int(sum(leaked.values()))
    rep.append(f"Visual near-duplicate rule: DINOv2 cosine >= {args.dino_threshold}, "
               f"or dHash Hamming <= {args.dhash_threshold} with cosine >= 0.70.")
    rep.append(f"Near-duplicate pairs found: {len(edges):,} "
               f"({int(edges['cross_domain'].sum()) if len(edges) else 0} cross-domain).")
    rep.append("")
    rep.append(domain_summary.to_markdown(index=False))
    rep.append("")
    rep.append(f"Reference images whose near-duplicate group spans another CV fold: "
               f"{n_leak} / {n_ref} ({100 * n_leak / max(n_ref, 1):.1f}%).")
    if cv_rows:
        cvm = pd.DataFrame(cv_rows).groupby("subset")[["macro_f1", "mAP", "macro_AUROC"]].agg(["mean", "std"])
        rep.append("")
        rep.append("Five-fold CV, all validation images vs. clean (leaked removed):")
        rep.append("")
        rep.append(cvm.round(4).to_markdown())
    rep.append("")
    rep.append("External evaluation, all vs. clean (images sharing a group with any reference image removed):")
    rep.append("")
    rep.append(ext.round(4).to_markdown(index=False))
    rep.append("")
    rep.append("Field provenance:")
    rep.append("")
    rep.append(prov[["provenance", "n_images"]].to_markdown(index=False))
    if not args.no_grouped_dataset:
        rep.append("")
        rep.append(f"Group-aware split written to `check1_leakage/paper1_frozen_dataset_v3_grouped/` "
                   f"({n_cross_fold_groups} reference groups were split across folds in v3). "
                   "Optional re-run:")
        rep.append("")
        rep.append("```")
        rep.append('python train_paper1_multilabel.py --model convnext_tiny '
                   f'--data-dir "{out / "paper1_frozen_dataset_v3_grouped"}" '
                   f'--results-dir "{BASE_DIR / "paper1_model_results_grouped"}"')
        rep.append("```")


# =============================================================================
# 4. CHECK 2 — UNIT OF MULTI-LABEL ATTRIBUTION
# =============================================================================

def lineage_from_json_path(path: str) -> str | None:
    for part in reversed(Path(path).parts[:-1]):
        m = re.match(r"^\s*(0[1-9]|1[0-2])(?!\d)", part)
        if m:
            return m.group(1)
    return None


def shape_bbox(shape: dict) -> Tuple[float, float, float, float] | None:
    pts = np.asarray(shape.get("points") or [], dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return None
    if shape.get("shape_type") == "circle" and len(pts) >= 2:
        r = float(np.linalg.norm(pts[1] - pts[0]))
        cx, cy = pts[0]
        return cx - r, cy - r, cx + r, cy + r
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def load_lineage_annotations(all_json_paths: str) -> Dict[str, dict]:
    """lineage id -> {shapes, width, height, signature} (largest JSON per lineage)."""
    out = {}
    for jp in str(all_json_paths).split(" || "):
        jp = jp.strip()
        sid = lineage_from_json_path(jp)
        if not sid or not Path(jp).exists():
            continue
        try:
            data = json.loads(Path(jp).read_text(encoding="utf-8"))
        except Exception:
            continue
        shapes = data.get("shapes") or []
        core = [s for s in shapes if re.match(r"^[A-E]\d+$", str(s.get("label", "")).strip())]
        shapes = core or shapes
        signature = tuple(sorted(
            (str(s.get("label")), json.dumps(s.get("points"))) for s in shapes
        ))
        record = {
            "shapes": shapes,
            "width": float(data.get("imageWidth") or 0),
            "height": float(data.get("imageHeight") or 0),
            "signature": signature,
        }
        if sid not in out or len(shapes) > len(out[sid]["shapes"]):
            out[sid] = record
    return out


def footprint_mask(record: dict, grid: int = 64) -> np.ndarray:
    mask = np.zeros((grid, grid), dtype=bool)
    w, h = record["width"], record["height"]
    if w <= 0 or h <= 0:
        return mask
    for shape in record["shapes"]:
        bb = shape_bbox(shape)
        if bb is None:
            continue
        x0 = int(np.clip(np.floor(bb[0] / w * grid), 0, grid - 1))
        y0 = int(np.clip(np.floor(bb[1] / h * grid), 0, grid - 1))
        x1 = int(np.clip(np.ceil(bb[2] / w * grid), x0 + 1, grid))
        y1 = int(np.clip(np.ceil(bb[3] / h * grid), y0 + 1, grid))
        mask[y0:y1, x0:x1] = True
    return mask


def classify_multilabel_row(row: pd.Series) -> dict:
    expected = [s for s in str(row["style_ids"]).split("|") if s]
    ann = load_lineage_annotations(row["all_json_paths"])
    found = [s for s in expected if s in ann]
    scene = bool(SCENE_PATTERN.search(row["stem"]))

    result = {
        "n_lineages": len(expected),
        "n_lineage_jsons_found": len(found),
        "scene_word_in_filename": int(scene),
        "identical_annotation_across_lineages": 0,
        "min_pair_overlap": np.nan,
        "max_pair_overlap": np.nan,
        "spatial_pattern": "",
        "heuristic_unit": "",
    }

    if len(found) < 2:
        result["spatial_pattern"] = "lineage_jsons_missing"
        result["heuristic_unit"] = "ambiguous"
        return result

    if any(len(ann[s]["shapes"]) == 0 for s in found):
        result["spatial_pattern"] = "no_geometry_for_some_lineage"
        result["heuristic_unit"] = "ambiguous"
        return result

    identical = all(ann[s]["signature"] == ann[found[0]]["signature"] for s in found)
    result["identical_annotation_across_lineages"] = int(identical)

    masks = {s: footprint_mask(ann[s]) for s in found}
    overlaps = []
    for a, b in combinations(found, 2):
        inter = np.logical_and(masks[a], masks[b]).sum()
        denom = max(min(masks[a].sum(), masks[b].sum()), 1)
        overlaps.append(inter / denom)
    result["min_pair_overlap"] = float(min(overlaps))
    result["max_pair_overlap"] = float(max(overlaps))

    if identical:
        pattern = "shared_annotation"
    elif min(overlaps) >= 0.5:
        pattern = "co_located"
    elif max(overlaps) <= 0.15:
        pattern = "spatially_separate"
    else:
        pattern = "partial_overlap"
    result["spatial_pattern"] = pattern

    # A JSON copied unchanged into several lineage folders means the lineages
    # were attributed to the image as a whole; geometry cannot tell whether
    # one or several buildings carry them, so these go to manual coding.
    if pattern == "shared_annotation":
        result["heuristic_unit"] = "image_level_unresolved"
    elif pattern == "spatially_separate":
        result["heuristic_unit"] = "likely_multiple_objects"
    elif pattern == "co_located" and not scene:
        result["heuristic_unit"] = "likely_single_building"
    else:
        result["heuristic_unit"] = "ambiguous"

    return result


def draw_lineage_overlay(row: pd.Series) -> Image.Image:
    image = open_rgb(row["image_path"])
    ann = load_lineage_annotations(row["all_json_paths"])
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 250)

    for k, sid in enumerate(s for s in str(row["style_ids"]).split("|") if s):
        if sid not in ann:
            continue
        rec = ann[sid]
        sx = image.width / rec["width"] if rec["width"] else 1.0
        sy = image.height / rec["height"] if rec["height"] else 1.0
        colour = LINEAGE_COLOURS[k % len(LINEAGE_COLOURS)]
        offset = 2 * k * width   # offsets keep identical boxes visible
        for shape in rec["shapes"]:
            bb = shape_bbox(shape)
            if bb is None:
                continue
            draw.rectangle(
                [bb[0] * sx + offset, bb[1] * sy + offset, bb[2] * sx + offset, bb[3] * sy + offset],
                outline=colour,
                width=width,
            )
    return image


def cooccurrence_jaccard(Y: np.ndarray) -> np.ndarray:
    M = np.eye(Y.shape[1])
    for i, j in PAIRS:
        a, b = Y[:, i].astype(bool), Y[:, j].astype(bool)
        union = np.logical_or(a, b).sum()
        M[i, j] = M[j, i] = np.logical_and(a, b).sum() / union if union else 0.0
    return M


def top_pairs(M: np.ndarray, k: int = 5) -> str:
    vals = upper_tri_values(M)
    order = np.argsort(-vals)[:k]
    return "; ".join(
        f"{STYLE_IDS[PAIRS[o][0]]}-{STYLE_IDS[PAIRS[o][1]]} {vals[o]:.3f}" for o in order
    )


def check_multilabel(ctx: Context) -> None:
    args = ctx.args
    out = ctx.out_dir / "check2_multilabel"
    out.mkdir(parents=True, exist_ok=True)
    rep = ctx.report.setdefault("multilabel", [])
    df, Y = ctx.df, ctx.Y

    ml = df[df["multi_lineage"] == 1].copy()
    print(f"[CHECK 2] Classifying {len(ml)} multi-label images ...")

    results = [classify_multilabel_row(r) for _, r in ml.iterrows()]
    ml = pd.concat([ml.reset_index(), pd.DataFrame(results)], axis=1).rename(columns={"index": "row"})

    keep = [
        "row", "domain", "pixel_sha256", "stem", "style_ids", "n_lineages",
        "n_lineage_jsons_found", "identical_annotation_across_lineages",
        "min_pair_overlap", "max_pair_overlap", "scene_word_in_filename",
        "spatial_pattern", "heuristic_unit", "image_path",
    ]
    save_csv(ml[keep], out / "10_multilabel_unit_heuristic.csv")

    summary = (
        ml.groupby(["domain", "heuristic_unit"]).size().unstack(fill_value=0)
        .reindex(columns=["likely_single_building", "ambiguous", "image_level_unresolved",
                          "likely_multiple_objects"], fill_value=0)
    )
    summary["total"] = summary.sum(axis=1)
    summary.to_csv(out / "11_multilabel_unit_summary.csv", encoding="utf-8-sig")
    pattern_summary = ml.groupby(["domain", "spatial_pattern"]).size().unstack(fill_value=0)
    pattern_summary.to_csv(out / "11b_spatial_pattern_summary.csv", encoding="utf-8-sig")

    # ---- stratified manual coding sample ------------------------------------
    # Unresolved strata get a larger sample: geometry says nothing about them.
    def stratum_n(unit: str) -> int:
        if unit in ("image_level_unresolved", "ambiguous"):
            return 3 * args.sample_per_stratum
        return args.sample_per_stratum

    sample = pd.concat(
        [
            g.sample(min(len(g), stratum_n(unit)), random_state=args.seed)
            for (_, unit), g in ml.groupby(["domain", "heuristic_unit"])
        ],
        ignore_index=True,
    )
    sample.insert(0, "sample_no", np.arange(1, len(sample) + 1))
    sample["manual_unit"] = ""
    sample["manual_notes"] = ""
    template = sample[[
        "sample_no", "domain", "stem", "style_ids", "spatial_pattern",
        "heuristic_unit", "manual_unit", "manual_notes", "image_path",
    ]]
    save_csv(template, out / "12_manual_coding_sample.csv")

    per_sheet = 24
    for start in range(0, len(sample), per_sheet):
        chunk = sample.iloc[start:start + per_sheet]
        tiles, caps = [], []
        for _, r in chunk.iterrows():
            tiles.append(draw_lineage_overlay(r))
            ids = [s for s in str(r["style_ids"]).split("|") if s]
            caps.append(f"#{r['sample_no']} {r['domain']} | {'+'.join(ids)}\n{r['heuristic_unit']}")
        save_contact_sheet(
            tiles, caps,
            out / f"13_manual_sample_sheet_{start // per_sheet + 1:02d}.jpg",
            4, (360, 300),
        )

    (out / "13_HOW_TO_CODE.txt").write_text(
        "Box colours follow the order of style_ids in the caption:\n"
        "1st = red, 2nd = blue, 3rd = green, 4th = orange.\n"
        "Boxes of later lineages are shifted slightly so identical boxes stay visible.\n\n"
        "Fill `manual_unit` in 12_manual_coding_sample.csv with one of:\n"
        "  single_building        one building carries all assigned lineages\n"
        "  multiple_buildings     the lineages belong to different buildings in the frame\n"
        "  building_plus_context  one building plus railway/infrastructure/landscape context\n"
        "  unclear\n\n"
        "Save as 12_manual_coding_sample_FILLED.csv and re-run:\n"
        "  python paper1_analysis_checks.py --checks multilabel\n",
        encoding="utf-8",
    )

    rep.append("Heuristic unit of multi-label attribution (from per-lineage LabelMe geometry):")
    rep.append("")
    rep.append(summary.to_markdown())
    rep.append("")
    rep.append("Spatial pattern of lineage annotations:")
    rep.append("")
    rep.append(pattern_summary.to_markdown())

    # ---- manual coding evaluation (if filled) --------------------------------
    filled = out / "12_manual_coding_sample_FILLED.csv"
    if filled.exists():
        coded = pd.read_csv(filled, encoding="utf-8-sig", keep_default_na=False)
        coded = coded[coded["manual_unit"].isin(MANUAL_CODES)]
        if len(coded):
            cross = pd.crosstab(coded["heuristic_unit"], coded["manual_unit"])
            cross.to_csv(out / "14_manual_vs_heuristic.csv", encoding="utf-8-sig")

            # Stratum-weighted estimate of manual proportions per domain.
            sizes = ml.groupby(["domain", "heuristic_unit"]).size()
            est_rows = []
            for domain in DOMAINS:
                cd = coded[coded["domain"] == domain]
                if cd.empty:
                    continue
                total = sizes.loc[domain].sum()
                est = {c: 0.0 for c in MANUAL_CODES}
                for stratum, g in cd.groupby("heuristic_unit"):
                    weight = sizes.get((domain, stratum), 0) / total
                    props = g["manual_unit"].value_counts(normalize=True)
                    for c in MANUAL_CODES:
                        est[c] += weight * props.get(c, 0.0)
                est_rows.append({"domain": domain, "n_coded": len(cd), "n_multilabel": int(total),
                                 **{f"est_prop_{c}": round(v, 3) for c, v in est.items()}})
            est_df = pd.DataFrame(est_rows)
            save_csv(est_df, out / "15_manual_weighted_estimate.csv")

            rep.append("")
            rep.append(f"Manual coding: {len(coded)} images coded.")
            rep.append("")
            rep.append(cross.to_markdown())
            rep.append("")
            rep.append("Stratum-weighted estimate of the multi-label unit:")
            rep.append("")
            rep.append(est_df.to_markdown(index=False))

    # ---- co-occurrence without likely multi-object images --------------------
    dino = load_square_matrix(args.dino_dir / "04_prototype_similarity_reference.csv")
    ad = published_ad_matrix(ctx)
    multi_rows = set(ml.loc[ml["heuristic_unit"] == "likely_multiple_objects", "row"])

    co_rows = []
    for domain in ["historical", "field"]:
        dmask = (df["domain"] == domain).to_numpy()
        restricted = dmask & ~np.isin(np.arange(len(df)), list(multi_rows))
        M_all = cooccurrence_jaccard(Y[dmask])
        M_res = cooccurrence_jaccard(Y[restricted])
        rho_same = safe_spearman(upper_tri_values(M_all), upper_tri_values(M_res))
        rd_all, pd_all = matrix_permutation_spearman(dino, M_all, args.permutations, args.seed)
        ra_all, pa_all = matrix_permutation_spearman(ad, M_all, args.permutations, args.seed + 1)
        rd_res, pd_res = matrix_permutation_spearman(dino, M_res, args.permutations, args.seed + 2)
        ra_res, pa_res = matrix_permutation_spearman(ad, M_res, args.permutations, args.seed + 3)
        co_rows.append({
            "domain": domain,
            "n_images_all": int(dmask.sum()),
            "n_images_restricted": int(restricted.sum()),
            "spearman_all_vs_restricted": rho_same,
            "top5_all": top_pairs(M_all),
            "top5_restricted": top_pairs(M_res),
            "DINO_vs_cooc_rho_all": rd_all, "DINO_vs_cooc_p_all": pd_all,
            "DINO_vs_cooc_rho_restricted": rd_res, "DINO_vs_cooc_p_restricted": pd_res,
            "AD_vs_cooc_rho_all": ra_all, "AD_vs_cooc_p_all": pa_all,
            "AD_vs_cooc_rho_restricted": ra_res, "AD_vs_cooc_p_restricted": pa_res,
        })
    co = pd.DataFrame(co_rows)
    save_csv(co, out / "16_cooccurrence_restricted.csv")

    rep.append("")
    rep.append("Observed co-occurrence with likely multi-object images removed:")
    rep.append("")
    rep.append(co.drop(columns=["top5_all", "top5_restricted"]).round(4).to_markdown(index=False))
    for _, r in co.iterrows():
        rep.append(f"- {r['domain']} top-5 all: {r['top5_all']}")
        rep.append(f"- {r['domain']} top-5 restricted: {r['top5_restricted']}")


# =============================================================================
# 5. CHECK 3 — ANNOTATION DENSITY / COUNTING UNIT
# =============================================================================

def transform_counts(X: np.ndarray, unit: str) -> np.ndarray:
    if unit == "instance":
        return X
    if unit == "presence":
        return (X > 0).astype(float)
    if unit == "image_normalised":
        totals = X.sum(axis=1, keepdims=True)
        return np.divide(X, totals, out=np.zeros_like(X), where=totals > 0)
    raise ValueError(unit)


def lineage_profiles(X: np.ndarray, Y: np.ndarray, rows: np.ndarray, alpha: float) -> np.ndarray:
    counts = np.vstack([X[rows[Y[rows, k] == 1]].sum(axis=0) for k in range(12)])
    return smoothed_profile(counts, alpha)


def prototypes_from_reference(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Copied from paper1_dinov2_crossdomain.py."""
    single = Y.sum(axis=1) == 1
    P = []
    for j in range(Y.shape[1]):
        mask = single & (Y[:, j] == 1)
        if mask.sum() < 5:
            mask = Y[:, j] == 1
        p = X[mask].mean(axis=0)
        P.append(p / max(np.linalg.norm(p), 1e-12))
    return np.stack(P, axis=0)


def published_ad_matrix(ctx: Context) -> np.ndarray:
    """
    A–D (69-symbol) instance-based matrix, recomputed exactly as in
    paper1_visual_genealogy_final.py. Recomputed rather than read from
    05_symbol_similarity_reference.csv, because the copy on disk may come
    from the earlier A–E (84-symbol) run.
    """
    df, Y = ctx.df, ctx.Y
    canonical = np.nonzero((df["domain"] == "reference").to_numpy() & (Y.sum(axis=1) == 1))[0]
    return cosine_similarity(lineage_profiles(ctx.X_ad, Y, canonical, ctx.args.alpha))


def summarise_quantiles(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return (
        float(np.median(values)),
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def check_density(ctx: Context) -> None:
    args = ctx.args
    out = ctx.out_dir / "check3_density"
    out.mkdir(parents=True, exist_ok=True)
    rep = ctx.report.setdefault("density", [])
    df, Y, X = ctx.df, ctx.Y, ctx.X_ad
    rng = np.random.default_rng(args.seed)

    ref = (df["domain"] == "reference").to_numpy()
    canonical = np.nonzero(ref & (Y.sum(axis=1) == 1))[0]
    dino = load_square_matrix(args.dino_dir / "04_prototype_similarity_reference.csv")
    ad_published = published_ad_matrix(ctx)
    published_regions = classify_regions(dino, ad_published)

    units = ["instance", "presence", "image_normalised"]
    matrices, profiles = {}, {}
    for unit in units:
        profiles[unit] = lineage_profiles(transform_counts(X, unit), Y, canonical, args.alpha)
        matrices[unit] = cosine_similarity(profiles[unit])

    # Check that the baseline reproduces the manuscript (Table 7).
    manuscript_values = {("03", "05"): 0.922, ("11", "12"): 0.981, ("01", "04"): 0.814, ("03", "10"): 0.335}
    reproduction = "; ".join(
        f"{a}-{b} {ad_published[STYLE_IDS.index(a), STYLE_IDS.index(b)]:.3f} (manuscript {v:.3f})"
        for (a, b), v in manuscript_values.items()
    )
    print(f"[CHECK 3] A–D baseline vs manuscript: {reproduction}")

    stale_note = ""
    stale_path = args.genealogy_dir / "05_symbol_similarity_reference.csv"
    if stale_path.exists():
        stale_diff = float(np.abs(load_square_matrix(stale_path) - ad_published).max())
        if stale_diff > 1e-6:
            stale_note = (f"Note: `{stale_path.name}` on disk differs from the A–D-only recomputation "
                          f"(max |diff| = {stale_diff:.3f}); it appears to be from the earlier A–E run. "
                          "Re-run paper1_visual_genealogy_final.py before archiving the supplement.")

    # ---- matrix-level comparison --------------------------------------------
    variant_rows, pair_rows = [], []
    for k, unit in enumerate(units):
        M = matrices[unit]
        rho_pub, p_pub = matrix_permutation_spearman(ad_published, M, args.permutations, args.seed + k)
        rho_dino, p_dino = matrix_permutation_spearman(dino, M, args.permutations, args.seed + 10 + k)
        regions = classify_regions(dino, M)
        region_counts = pd.Series(regions).value_counts()
        variant_rows.append({
            "counting_unit": unit,
            "spearman_with_published_AD": rho_pub,
            "permutation_p_with_published_AD": p_pub,
            "spearman_with_DINOv2": rho_dino,
            "permutation_p_with_DINOv2": p_dino,
            "AD_median": float(np.median(upper_tri_values(M))),
            "region_agreement_with_published": float(np.mean(np.array(regions) == np.array(published_regions))),
            "region_kappa_with_published": float(cohen_kappa_score(published_regions, regions)),
            **{f"n_{r}": int(region_counts.get(r, 0)) for r in REGION_LABELS},
        })

        vals = upper_tri_values(M)
        ranks = rankdata(-vals, method="min")
        for idx, (i, j) in enumerate(PAIRS):
            pair_rows.append({
                "counting_unit": unit,
                "pair": f"{STYLE_IDS[i]}-{STYLE_IDS[j]}",
                "style_1": STYLE_NAMES_EN[STYLE_IDS[i]],
                "style_2": STYLE_NAMES_EN[STYLE_IDS[j]],
                "DINOv2_similarity": float(dino[i, j]),
                "AD_similarity": float(vals[idx]),
                "AD_rank": int(ranks[idx]),
                "region": regions[idx],
                "published_region": published_regions[idx],
            })

    variants = pd.DataFrame(variant_rows)
    pairs = pd.DataFrame(pair_rows)
    save_csv(variants, out / "20_counting_unit_matrix_comparison.csv")
    save_csv(pairs, out / "21_counting_unit_all_pairs.csv")

    key = pairs[pairs["pair"].isin([f"{a}-{b}" for a, b in KEY_PAIRS])]
    key_wide = key.pivot_table(index="pair", columns="counting_unit", values=["AD_similarity", "AD_rank"])
    key_wide.to_csv(out / "22_key_pairs.csv", encoding="utf-8-sig")
    key_regions = key.pivot_table(index="pair", columns="counting_unit", values="region", aggfunc="first")

    # ---- density diagnostics -------------------------------------------------
    instances = X.sum(axis=1)
    types = (X > 0).sum(axis=1)
    diag_rows = []
    for k, sid in enumerate(STYLE_IDS):
        rows = canonical[Y[canonical, k] == 1]
        row = {
            "style_id": sid,
            "style_name_en": STYLE_NAMES_EN[sid],
            "n_canonical_images": len(rows),
            "mean_AD_instances_per_image": float(instances[rows].mean()),
            "mean_AD_types_per_image": float(types[rows].mean()),
            "share_images_without_AD": float((instances[rows] == 0).mean()),
        }
        for unit in units:
            M = matrices[unit]
            row[f"mean_offdiag_similarity_{unit}"] = float((M[k].sum() - 1) / 11)
        diag_rows.append(row)
    diag = pd.DataFrame(diag_rows)
    save_csv(diag, out / "23_lineage_density_diagnostics.csv")

    rho_density = {
        unit: safe_spearman(diag["mean_AD_instances_per_image"], diag[f"mean_offdiag_similarity_{unit}"])
        for unit in units
    }

    domain_density = []
    for domain in DOMAINS:
        d = (df["domain"] == domain).to_numpy()
        domain_density.append({
            "domain": domain,
            "n_images": int(d.sum()),
            "mean_AD_instances_per_image": float(instances[d].mean()),
            "median_AD_instances_per_image": float(np.median(instances[d])),
            "mean_AD_types_per_image": float(types[d].mean()),
            "share_images_without_AD": float((instances[d] == 0).mean()),
        })
    domain_density = pd.DataFrame(domain_density)
    save_csv(domain_density, out / "24_domain_density.csv")

    # ---- A–D continuity by counting unit ------------------------------------
    cont_rows = []
    for unit in units:
        Xu = transform_counts(X, unit)
        for domain in ["historical", "field"]:
            d = (df["domain"] == domain).to_numpy()
            for k, sid in enumerate(STYLE_IDS):
                rows = np.nonzero(d & (Y[:, k] == 1))[0]
                if len(rows) == 0:
                    continue
                target = smoothed_profile(Xu[rows].sum(axis=0), args.alpha)
                js = float(jensenshannon(profiles[unit][k], target, base=2.0))
                cont_rows.append({
                    "counting_unit": unit,
                    "target_domain": domain,
                    "style_id": sid,
                    "style_name_en": STYLE_NAMES_EN[sid],
                    "n_target_positive": len(rows),
                    "adequate_support": int(len(rows) >= MIN_DOMAIN_LINEAGE_N),
                    "AD_continuity": 1.0 - js,
                })
    cont = pd.DataFrame(cont_rows)
    save_csv(cont, out / "25_AD_continuity_by_counting_unit.csv")
    cont_wide = cont.pivot_table(index=["target_domain", "style_id"], columns="counting_unit", values="AD_continuity")
    cont_adequate = cont[cont["adequate_support"] == 1].groupby(["counting_unit", "target_domain"])["AD_continuity"].mean().unstack()
    cont_rho = {
        unit: safe_spearman(cont_wide["instance"], cont_wide[unit]) for unit in units
    }

    # ---- equal-n subsampling --------------------------------------------------
    per_lineage = [canonical[Y[canonical, k] == 1] for k in range(12)]
    n_min = min(len(r) for r in per_lineage)
    print(f"[CHECK 3] Equal-n subsampling: {args.n_subsample} reps x {n_min} images per lineage ...")
    sub = {unit: {"rho_dino": [], "rho_published": []} for unit in ["instance", "presence"]}
    for _ in range(args.n_subsample):
        rows = np.concatenate([rng.choice(r, n_min, replace=False) for r in per_lineage])
        for unit in sub:
            M = cosine_similarity(lineage_profiles(transform_counts(X, unit), Y, rows, args.alpha))
            sub[unit]["rho_dino"].append(safe_spearman(upper_tri_values(dino), upper_tri_values(M)))
            sub[unit]["rho_published"].append(safe_spearman(upper_tri_values(ad_published), upper_tri_values(M)))

    sub_rows = []
    for unit, d in sub.items():
        for stat, vals in d.items():
            med, lo, hi = summarise_quantiles(vals)
            sub_rows.append({"counting_unit": unit, "statistic": stat, "n_per_lineage": n_min,
                             "median": med, "q2.5": lo, "q97.5": hi})
    sub_df = pd.DataFrame(sub_rows)
    save_csv(sub_df, out / "26_equal_n_subsampling.csv")

    # ---- joint bootstrap of the two-layer correlation (C14) -----------------
    P_full = prototypes_from_reference(ctx.emb[ref], Y[ref])
    dino_error = float(np.abs(P_full @ P_full.T - dino).max())
    print(f"[CHECK 3] Recomputed DINOv2 prototype matrix vs published file: max |diff| = {dino_error:.2e}")

    print(f"[CHECK 3] Joint bootstrap of the two-layer correlation: {args.n_boot} reps ...")
    boot = {unit: [] for unit in ["instance", "presence"]}
    for _ in range(args.n_boot):
        rows_by_k = [rng.choice(r, len(r), replace=True) for r in per_lineage]
        P = np.stack([
            (lambda p: p / max(np.linalg.norm(p), 1e-12))(ctx.emb[r].mean(axis=0))
            for r in rows_by_k
        ])
        D = P @ P.T
        for unit in boot:
            Xu = transform_counts(X, unit)
            counts = np.vstack([Xu[r].sum(axis=0) for r in rows_by_k])
            M = cosine_similarity(smoothed_profile(counts, args.alpha))
            boot[unit].append(safe_spearman(upper_tri_values(D), upper_tri_values(M)))

    boot_rows = []
    for unit, vals in boot.items():
        vals = np.array(vals)
        med, lo, hi = summarise_quantiles(vals)
        boot_rows.append({
            "counting_unit": unit,
            "observed_rho": float(variants.loc[variants["counting_unit"] == unit, "spearman_with_DINOv2"].iloc[0]),
            "bootstrap_median": med,
            "ci95_low": lo,
            "ci95_high": hi,
            "share_rho_gt_0": float(np.mean(vals > 0)),
        })
    boot_df = pd.DataFrame(boot_rows)
    save_csv(boot_df, out / "27_two_layer_rho_bootstrap.csv")

    # ---- report --------------------------------------------------------------
    rep.append(f"Sanity: A–D baseline (instance, alpha = {args.alpha}) — {reproduction}. "
               f"Recomputed DINOv2 prototypes reproduce the published DINOv2 matrix "
               f"(max |diff| = {dino_error:.1e}).")
    if stale_note:
        rep.append("")
        rep.append(stale_note)
    rep.append("")
    rep.append(variants.round(4).to_markdown(index=False))
    rep.append("")
    rep.append("Key manuscript pairs — region by counting unit:")
    rep.append("")
    rep.append(key_regions.to_markdown())
    rep.append("")
    rep.append("Key manuscript pairs — A–D similarity (rank of 66) by counting unit:")
    rep.append("")
    rep.append(key_wide.round(3).to_markdown())
    rep.append("")
    rep.append("Lineage annotation density (canonical single-lineage reference images):")
    rep.append("")
    rep.append(diag.round(3).to_markdown(index=False))
    rep.append("")
    rep.append("Spearman (n = 12) between lineage mean A–D instances/image and mean off-diagonal similarity: "
               + ", ".join(f"{u} = {v:.3f}" for u, v in rho_density.items()))
    rep.append("")
    rep.append("Annotation density by domain:")
    rep.append("")
    rep.append(domain_density.round(3).to_markdown(index=False))
    rep.append("")
    rep.append("Mean A–D continuity (adequate support only) by counting unit:")
    rep.append("")
    rep.append(cont_adequate.round(3).to_markdown())
    rep.append("")
    rep.append("Spearman of lineage-level continuity with the instance version: "
               + ", ".join(f"{u} = {v:.3f}" for u, v in cont_rho.items()))
    rep.append("")
    rep.append(f"Equal-n subsampling ({n_min} images per lineage, {args.n_subsample} reps):")
    rep.append("")
    rep.append(sub_df.round(3).to_markdown(index=False))
    rep.append("")
    rep.append(f"Joint bootstrap of DINOv2–A–D rho ({args.n_boot} reps, images resampled within lineage, "
               "both layers recomputed):")
    rep.append("")
    rep.append(boot_df.round(3).to_markdown(index=False))


# =============================================================================
# 6. CHECK 4 — PAIRED BOOTSTRAP OF EXTERNAL PERFORMANCE
# =============================================================================

def load_convnext_external(path: Path, shas: pd.Series) -> Tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        print(f"  [WARN] ConvNeXt predictions not found: {path}")
        return None
    pred = pd.read_csv(path, encoding="utf-8-sig").set_index("pixel_sha256")
    missing = set(shas) - set(pred.index)
    if missing:
        raise RuntimeError(f"{len(missing)} images missing from {path.name}")
    pred = pred.loc[list(shas)]
    probs = pred[[f"p_{s}" for s in STYLE_IDS]].to_numpy(float)
    binary = pred[[f"pred_{s}" for s in STYLE_IDS]].to_numpy(int)
    return probs, binary


def fit_linear_probe(Xr: np.ndarray, Yr: np.ndarray):
    """Same configuration as paper1_dinov2_crossdomain.py."""
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
    return clf


def bootstrap_p(diff: np.ndarray) -> float:
    diff = diff[np.isfinite(diff)]
    b = len(diff)
    tail = min((diff <= 0).sum(), (diff >= 0).sum())
    return float(min(1.0, 2 * (tail + 1) / (b + 1)))


def check_bootstrap(ctx: Context) -> None:
    args = ctx.args
    out = ctx.out_dir / "check4_bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    rep = ctx.report.setdefault("bootstrap", [])
    df, Y = ctx.df, ctx.Y
    rng = np.random.default_rng(args.seed)

    groups = build_groups(ctx)
    ref = (df["domain"] == "reference").to_numpy()
    P = prototypes_from_reference(ctx.emb[ref], Y[ref])

    probe = None
    if not args.skip_linear_probe:
        print("[CHECK 4] Refitting the frozen DINOv2 linear probe on reference embeddings ...")
        probe = fit_linear_probe(ctx.emb[ref], Y[ref])

    summary_rows, diff_rows, lineage_rows, sanity = [], [], [], []

    for domain, pred_name in [
        ("historical", "historical_external_ensemble_predictions.csv"),
        ("field", "field_external_ensemble_predictions.csv"),
    ]:
        rows = np.nonzero((df["domain"] == domain).to_numpy())[0]
        Yd = Y[rows]
        scores = {}
        conv = load_convnext_external(args.model_dir / pred_name, df.loc[rows, "pixel_sha256"])
        if conv is not None:
            scores["ConvNeXt-Tiny"] = conv[0]
        scores["DINOv2 prototype"] = ctx.emb[rows] @ P.T
        if probe is not None:
            scores["DINOv2 linear probe"] = probe.predict_proba(ctx.emb[rows])

        # sanity: fast metrics equal sklearn on the full sample
        for name, S in scores.items():
            sk_auc = np.nanmean([
                roc_auc_score(Yd[:, k], S[:, k]) if len(np.unique(Yd[:, k])) == 2 else np.nan
                for k in range(12)
            ])
            sk_ap = np.nanmean([
                average_precision_score(Yd[:, k], S[:, k]) if Yd[:, k].sum() else np.nan
                for k in range(12)
            ])
            fm = ranking_metrics(Yd, S)
            sanity.append(f"{domain} {name}: macro-AUROC {fm['macro_AUROC']:.4f} (sklearn {sk_auc:.4f}), "
                          f"mAP {fm['mAP']:.4f} (sklearn {sk_ap:.4f})")

        clusters = [
            np.nonzero(groups.loc[rows, "group_id"].to_numpy() == g)[0]
            for g in np.unique(groups.loc[rows, "group_id"])
        ]

        methods = list(scores)
        comparisons = [
            (a, b)
            for a, b in [
                ("DINOv2 prototype", "ConvNeXt-Tiny"),
                ("DINOv2 prototype", "DINOv2 linear probe"),
                ("DINOv2 linear probe", "ConvNeXt-Tiny"),
            ]
            if a in scores and b in scores
        ]

        for level in ["image", "cluster"]:
            print(f"[CHECK 4] {domain}: {level}-level bootstrap, {args.n_boot} reps "
                  f"({len(rows)} images, {len(clusters)} clusters) ...")
            boot = {m: {"macro_AUROC": [], "mAP": []} for m in methods}
            boot_f1 = []
            per_lineage = {m: [] for m in methods}

            for _ in range(args.n_boot):
                if level == "image":
                    idx = rng.integers(0, len(rows), len(rows))
                else:
                    pick = rng.integers(0, len(clusters), len(clusters))
                    idx = np.concatenate([clusters[p] for p in pick])
                Yb = Yd[idx]
                for m in methods:
                    aucs, aps = per_class_scores(Yb, scores[m][idx])
                    boot[m]["macro_AUROC"].append(np.nanmean(aucs))
                    boot[m]["mAP"].append(np.nanmean(aps))
                    if level == "image":
                        per_lineage[m].append(aucs)
                if conv is not None:
                    boot_f1.append(f1_score(Yb, conv[1][idx], average="macro", zero_division=0))

            for m in methods:
                point = ranking_metrics(Yd, scores[m])
                for metric in ["macro_AUROC", "mAP"]:
                    vals = np.array(boot[m][metric])
                    _, lo, hi = summarise_quantiles(vals)
                    summary_rows.append({
                        "domain": domain, "level": level, "method": m, "metric": metric,
                        "point": point[metric], "ci95_low": lo, "ci95_high": hi,
                    })
            if conv is not None:
                _, lo, hi = summarise_quantiles(boot_f1)
                summary_rows.append({
                    "domain": domain, "level": level, "method": "ConvNeXt-Tiny", "metric": "macro_F1",
                    "point": threshold_metrics(Yd, conv[1])["macro_f1"], "ci95_low": lo, "ci95_high": hi,
                })

            for a, b in comparisons:
                for metric in ["macro_AUROC", "mAP"]:
                    d = np.array(boot[a][metric]) - np.array(boot[b][metric])
                    pa, pb = ranking_metrics(Yd, scores[a])[metric], ranking_metrics(Yd, scores[b])[metric]
                    _, lo, hi = summarise_quantiles(d)
                    diff_rows.append({
                        "domain": domain, "level": level, "comparison": f"{a} − {b}", "metric": metric,
                        "point_difference": pa - pb, "ci95_low": lo, "ci95_high": hi,
                        "share_difference_gt_0": float(np.mean(d > 0)),
                        "bootstrap_p_two_sided": bootstrap_p(d),
                    })

            if level == "image" and "ConvNeXt-Tiny" in scores:
                A = np.vstack(per_lineage["DINOv2 prototype"])
                B = np.vstack(per_lineage["ConvNeXt-Tiny"])
                pa, _ = per_class_scores(Yd, scores["DINOv2 prototype"])
                pb, _ = per_class_scores(Yd, scores["ConvNeXt-Tiny"])
                for k, sid in enumerate(STYLE_IDS):
                    d = A[:, k] - B[:, k]
                    if not np.isfinite(d).any():
                        continue
                    _, lo, hi = summarise_quantiles(d)
                    lineage_rows.append({
                        "domain": domain, "style_id": sid, "style_name_en": STYLE_NAMES_EN[sid],
                        "n_positive": int(Yd[:, k].sum()),
                        "AUROC_DINOv2_prototype": pa[k], "AUROC_ConvNeXt": pb[k],
                        "difference": pa[k] - pb[k], "ci95_low": lo, "ci95_high": hi,
                        "ci_excludes_0": int(lo > 0 or hi < 0),
                    })

    summary = pd.DataFrame(summary_rows)
    diffs = pd.DataFrame(diff_rows)
    lineage = pd.DataFrame(lineage_rows)
    save_csv(summary, out / "30_external_bootstrap_summary.csv")
    save_csv(diffs, out / "31_external_bootstrap_differences.csv")
    if len(lineage):
        save_csv(lineage, out / "32_per_lineage_auroc_difference.csv")

    rep.append("Sanity (fast metrics vs sklearn on the full sample):")
    rep.extend(f"- {s}" for s in sanity)
    rep.append("")
    rep.append(f"95% percentile CIs, {args.n_boot} bootstrap replicates. "
               "`cluster` resamples building/site groups from CHECK 1 instead of single images.")
    rep.append("")
    rep.append(summary.round(4).to_markdown(index=False))
    rep.append("")
    rep.append("Paired differences:")
    rep.append("")
    rep.append(diffs.round(4).to_markdown(index=False))
    if len(lineage):
        rep.append("")
        rep.append("Per-lineage AUROC difference, DINOv2 prototype − ConvNeXt (image-level):")
        rep.append("")
        rep.append(lineage.round(3).to_markdown(index=False))


# =============================================================================
# 7. REPORT + MAIN
# =============================================================================

REPORT_TITLES = {
    "leakage": "Check 1 — Building/site grouping and leakage (C4, C8)",
    "multilabel": "Check 2 — Unit of multi-label attribution (C6)",
    "density": "Check 3 — Annotation density and counting unit (C11, C14)",
    "bootstrap": "Check 4 — Paired bootstrap of external performance (C13)",
}


def write_report(ctx: Context) -> Path:
    lines = [
        "# Paper 1 — analysis checks for the advisor review",
        "",
        f"Data: `{ctx.args.data_dir}`  ",
        f"Seed: {ctx.args.seed}",
        "",
    ]
    for key in ["leakage", "multilabel", "density", "bootstrap"]:
        if key in ctx.report:
            lines.append(f"## {REPORT_TITLES[key]}")
            lines.append("")
            lines.extend(ctx.report[key])
            lines.append("")
    path = ctx.out_dir / "00_analysis_checks_report.md"
    if path.exists():
        # keep sections from earlier partial runs
        old = path.read_text(encoding="utf-8")
        for key, title in REPORT_TITLES.items():
            if key not in ctx.report and f"## {title}" in old:
                section = old.split(f"## {title}", 1)[1]
                nxt = re.search(r"\n## ", section)
                lines.append(f"## {title}" + (section[: nxt.start()] if nxt else section).rstrip())
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness checks for Paper 1 (advisor review).")
    parser.add_argument("--checks", nargs="+", default=["all"],
                        choices=["all", "leakage", "multilabel", "density", "bootstrap"])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dino-dir", type=Path, default=DEFAULT_DINO_DIR)
    parser.add_argument("--genealogy-dir", type=Path, default=DEFAULT_GENEALOGY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dino-threshold", type=float, default=0.90,
                        help="DINOv2 cosine for a visual near-duplicate edge")
    parser.add_argument("--dhash-threshold", type=int, default=6,
                        help="dHash Hamming distance for a near-duplicate edge")
    parser.add_argument("--img-seq-gap", type=int, default=5,
                        help="max gap between IMG_xxxx numbers within one shooting sequence")
    parser.add_argument("--sample-per-stratum", type=int, default=15)
    parser.add_argument("--max-sheet-pairs", type=int, default=60)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-subsample", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=DIRICHLET_ALPHA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-linear-probe", action="store_true")
    parser.add_argument("--no-grouped-dataset", action="store_true")
    args = parser.parse_args()

    checks = ["leakage", "multilabel", "density", "bootstrap"] if "all" in args.checks else args.checks
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PAPER 1 — ANALYSIS CHECKS")
    print("=" * 100)
    print(f"Checks: {', '.join(checks)}")
    print(f"Output: {args.out_dir}\n")

    ctx = load_context(args)

    runners = {
        "leakage": check_leakage,
        "multilabel": check_multilabel,
        "density": check_density,
        "bootstrap": check_bootstrap,
    }
    for name in checks:
        print("-" * 100)
        print(f"Running {name} ...")
        runners[name](ctx)

    report = write_report(ctx)
    print("=" * 100)
    print("COMPLETE")
    print(f"Report: {report}")
    print("=" * 100)


if __name__ == "__main__":
    main()
