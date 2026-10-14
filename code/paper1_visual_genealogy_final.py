# -*- coding: utf-8 -*-
r"""
paper1_visual_genealogy_final.py

Paper 1 — FINAL visual-genealogy analysis
=========================================

This script is the publication-aligned replacement for:
    paper1_stage4_visual_genealogy.py

FINAL MANUSCRIPT RULES
----------------------
1. Main architectural genealogy uses ONLY A–D morphology symbols:
      A = Roof and upper-profile elements          (12)
      B = Openings                                 (16)
      C = Materials and structural elements        (20)
      D = Façade decoration and motifs             (21)
   Total main vocabulary = 69 symbols.

2. E1–E15 railway-infrastructural symbols NEVER enter:
      - A–D lineage profiles
      - A–D similarity matrix
      - DINOv2-vs-symbol matrix comparison
      - cross-domain symbol continuity
   E is exported separately only as supplementary/contextual information.

3. VHI is completely removed.
   This script does NOT calculate:
      - Visual Lineage Hybridization Index
      - VHI validation
      - per-image symbol-lineage evidence

4. Canonical A–D profiles are estimated ONLY from:
      reference-domain + single-lineage images.

5. Historical and field images never redefine canonical reference profiles.

6. Historical and field domains remain external observational domains.

7. Low-support lineage/domain continuity estimates (N < 15) are retained
   but explicitly flagged.

RUN AFTER
---------
1) repair_paper1_freeze_v3.py
2) paper1_dinov2_crossdomain.py

INPUTS
------
<PROJECT_ROOT>/paper1_frozen_dataset_v3/
    01_unique_image_multilabel_master.csv

<PROJECT_ROOT>/paper1_dinov2_results/
    04_prototype_similarity_reference.csv

OUTPUTS
-------
<PROJECT_ROOT>/paper1_visual_genealogy_results/

    00_stage4_summary.json
    01_symbol_family_summary.csv
    02_reference_symbol_profiles.csv
    03_reference_symbol_log_enrichment.csv
    04_signature_symbols_top10.csv
    05_symbol_similarity_reference.csv
    06_dinov2_symbol_similarity_convergence.csv
    07_lineage_cooccurrence_jaccard_reference.csv
    07_lineage_cooccurrence_jaccard_historical.csv
    07_lineage_cooccurrence_jaccard_field.csv
    08_similarity_convergence_all.csv
    09_symbol_continuity_by_lineage.csv
    10_pairwise_two_layer_genealogy.csv
    11_pairwise_cooccurrence_complete.csv

Supplementary/context-only E outputs:
    12_E_context_reference_log_enrichment.csv
    13_E_context_reference_profiles.csv

LEGACY VHI FILES ARE REMOVED FROM THE OUTPUT DIRECTORY IF PRESENT.

Dependencies
------------
pip install pandas numpy scipy scikit-learn

Recommended command
-------------------
python paper1_visual_genealogy_final.py
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 0. CONFIG
# =============================================================================

DEFAULT_DATA_DIR = Path(
    str(PROJECT_ROOT / "paper1_frozen_dataset_v3")
)
DEFAULT_DINO_DIR = Path(
    str(PROJECT_ROOT / "paper1_dinov2_results")
)
DEFAULT_OUT_DIR = Path(
    str(PROJECT_ROOT / "paper1_visual_genealogy_results")
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

# FINAL MAIN ARCHITECTURAL MORPHOLOGY VOCABULARY: A–D ONLY = 69 symbols.
AD_SYMBOLS = (
    [f"A{i}" for i in range(1, 13)]
    + [f"B{i}" for i in range(1, 17)]
    + [f"C{i}" for i in range(1, 21)]
    + [f"D{i}" for i in range(1, 22)]
)

# Supplementary/context only. NEVER included in main genealogy.
E_SYMBOLS = [f"E{i}" for i in range(1, 16)]

ALL_KNOWN_SYMBOLS = AD_SYMBOLS + E_SYMBOLS

FAMILY_NAMES = {
    "A": "Roof and upper-profile elements",
    "B": "Openings",
    "C": "Materials and structural elements",
    "D": "Façade decoration and motifs",
    "E": "Railway-infrastructural context",
}

MIN_DOMAIN_LINEAGE_N = 15
DIRICHLET_ALPHA = 0.5
PERMUTATIONS = 10000
SEED = 20260917

VALID_ROLES = {
    "development_reference",
    "external_historical",
    "external_field",
}

ROLE_TO_DOMAIN = {
    "development_reference": "reference",
    "external_historical": "historical",
    "external_field": "field",
}

# Remove these old outputs so the final folder cannot accidentally retain VHI files.
LEGACY_VHI_FILES = [
    "10_image_vhi.csv",
    "11_vhi_validation.csv",
    "12_lineage_evidence_by_image.csv",
]


# =============================================================================
# 1. BASIC HELPERS
# =============================================================================

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return path


def remove_legacy_vhi_outputs(out_dir: Path) -> None:
    """Delete legacy VHI artifacts if they remain from an earlier run."""
    removed = []
    for name in LEGACY_VHI_FILES:
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)

    if removed:
        print("[CLEANUP] Removed legacy VHI outputs:")
        for name in removed:
            print(f"          - {name}")


def parse_counts(value) -> Dict[str, int]:
    """
    Parse symbol_counts_json while retaining only known A–E codes.
    Unrecognized labels are ignored here because annotation harmonization
    is performed upstream.
    """
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


def count_vector(value, symbols: List[str]) -> np.ndarray:
    d = parse_counts(value)
    return np.array(
        [d.get(sym, 0) for sym in symbols],
        dtype=float,
    )


def style_vector(row: pd.Series) -> np.ndarray:
    return np.array(
        [int(float(row[f"y_{sid}"])) for sid in STYLE_IDS],
        dtype=int,
    )


def matrix_to_csv(
    matrix: np.ndarray,
    labels: List[str],
    path: Path,
) -> None:
    pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    ).to_csv(
        path,
        encoding="utf-8-sig",
    )


def upper_tri_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    result = spearmanr(x, y)
    # scipy compatibility
    rho = getattr(result, "statistic", result[0])
    return float(rho)


def matrix_permutation_spearman(
    A: np.ndarray,
    B: np.ndarray,
    n_perm: int = PERMUTATIONS,
    seed: int = SEED,
) -> Tuple[float, float]:
    """
    Matrix-concordance test used in the manuscript.

    Observed statistic:
        Spearman correlation between unique off-diagonal pairwise values.

    Null:
        symmetrically permute lineage labels of B.

    Returns
    -------
    observed_rho, two_sided_permutation_p
    """
    if A.shape != B.shape:
        raise ValueError(
            f"Matrix shapes differ: {A.shape} vs {B.shape}"
        )

    a = upper_tri_values(A)
    b = upper_tri_values(B)

    observed = safe_spearman(a, b)

    rng = np.random.default_rng(seed)
    hits = 0
    n = B.shape[0]

    for _ in range(n_perm):
        perm = rng.permutation(n)
        Bp = B[np.ix_(perm, perm)]
        rp = safe_spearman(a, upper_tri_values(Bp))

        if abs(rp) >= abs(observed):
            hits += 1

    p = (hits + 1) / (n_perm + 1)

    return observed, float(p)


def style_id_from_matrix_label(label: str) -> str | None:
    """
    Extract leading style ID from labels such as:
      01_俄罗斯传统建筑
      01 Russian Traditional
      01
    """
    m = re.match(r"^\s*(0[1-9]|1[0-2])(?=$|[^0-9])", str(label))
    return m.group(1) if m else None


def load_dino_similarity(path: Path) -> np.ndarray:
    """
    Load and robustly reorder the DINOv2 12x12 prototype similarity matrix
    to STYLE_IDS = 01...12.
    """
    df = pd.read_csv(
        require(path),
        index_col=0,
        encoding="utf-8-sig",
    )

    if df.shape != (12, 12):
        raise RuntimeError(
            f"Expected 12x12 DINO matrix, got {df.shape}: {path}"
        )

    row_map = {
        style_id_from_matrix_label(idx): idx
        for idx in df.index
    }
    col_map = {
        style_id_from_matrix_label(col): col
        for col in df.columns
    }

    if (
        all(sid in row_map for sid in STYLE_IDS)
        and all(sid in col_map for sid in STYLE_IDS)
    ):
        df = df.loc[
            [row_map[sid] for sid in STYLE_IDS],
            [col_map[sid] for sid in STYLE_IDS],
        ]

    matrix = df.to_numpy(dtype=float)

    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("DINO similarity matrix contains non-finite values.")

    return matrix


# =============================================================================
# 2. PROFILE / ENRICHMENT HELPERS
# =============================================================================

def smoothed_profile(
    counts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Convert counts into row-normalized smoothed probability profiles.
    """
    counts = np.asarray(counts, dtype=float)

    if counts.ndim == 1:
        return (
            counts + alpha
        ) / (
            counts.sum() + alpha * counts.shape[0]
        )

    return (
        counts + alpha
    ) / (
        counts.sum(axis=1, keepdims=True)
        + alpha * counts.shape[1]
    )


def build_reference_counts_and_profiles(
    X: np.ndarray,
    Y: np.ndarray,
    domain: np.ndarray,
    symbols: List[str],
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Canonical rule:
      reference-domain AND single-lineage images only.
    """
    reference_mask = (domain == "reference")
    single_mask = (Y.sum(axis=1) == 1)
    canonical_mask = reference_mask & single_mask

    counts = np.zeros(
        (len(STYLE_IDS), len(symbols)),
        dtype=float,
    )
    n_images = np.zeros(
        len(STYLE_IDS),
        dtype=int,
    )

    for k in range(len(STYLE_IDS)):
        m = canonical_mask & (Y[:, k] == 1)
        n_images[k] = int(m.sum())
        counts[k] = X[m].sum(axis=0)

    missing = [
        STYLE_IDS[k]
        for k, n in enumerate(n_images)
        if n == 0
    ]

    if missing:
        raise RuntimeError(
            "No single-lineage canonical reference images for: "
            + ", ".join(missing)
        )

    profiles = smoothed_profile(
        counts,
        alpha=alpha,
    )

    return counts, profiles, n_images


def export_profiles(
    path: Path,
    symbols: List[str],
    counts: np.ndarray,
    profiles: np.ndarray,
    n_images: np.ndarray,
    analysis_role: str,
) -> None:
    rows = []

    for k, sid in enumerate(STYLE_IDS):
        for s, symbol in enumerate(symbols):
            rows.append({
                "style_id": sid,
                "style_name": STYLE_NAMES[sid],
                "style_name_en": STYLE_NAMES_EN[sid],
                "canonical_single_lineage_images": int(n_images[k]),
                "symbol_code": symbol,
                "symbol_family": symbol[0],
                "analysis_role": analysis_role,
                "raw_count": int(counts[k, s]),
                "profile_probability": float(profiles[k, s]),
            })

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def compute_log_enrichment(
    counts: np.ndarray,
    profiles: np.ndarray,
    alpha: float,
) -> np.ndarray:
    global_counts = counts.sum(axis=0)
    global_profile = smoothed_profile(
        global_counts,
        alpha=alpha,
    )

    return np.log(
        profiles / global_profile[None, :]
    )


def export_log_enrichment(
    path: Path,
    symbols: List[str],
    counts: np.ndarray,
    log_enrichment: np.ndarray,
    analysis_role: str,
) -> None:
    rows = []

    for k, sid in enumerate(STYLE_IDS):
        for s, symbol in enumerate(symbols):
            rows.append({
                "style_id": sid,
                "style_name": STYLE_NAMES[sid],
                "style_name_en": STYLE_NAMES_EN[sid],
                "symbol_code": symbol,
                "symbol_family": symbol[0],
                "analysis_role": analysis_role,
                "raw_count": int(counts[k, s]),
                "log_enrichment": float(log_enrichment[k, s]),
            })

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def export_top_signatures(
    path: Path,
    symbols: List[str],
    counts: np.ndarray,
    log_enrichment: np.ndarray,
    top_n: int = 10,
    minimum_count: int = 5,
) -> None:
    """
    Statistical signatures, not essential/unique defining features.

    Ranking rule preserved from the original development script:
    prefer symbols with >= minimum_count instances where possible.
    """
    rows = []

    for k, sid in enumerate(STYLE_IDS):
        candidates = [
            s
            for s in range(len(symbols))
            if counts[k, s] >= minimum_count
        ]

        if len(candidates) < top_n:
            candidates = list(range(len(symbols)))

        top = sorted(
            candidates,
            key=lambda s: log_enrichment[k, s],
            reverse=True,
        )[:top_n]

        for rank, s in enumerate(top, 1):
            rows.append({
                "style_id": sid,
                "style_name": STYLE_NAMES[sid],
                "style_name_en": STYLE_NAMES_EN[sid],
                "rank": rank,
                "symbol_code": symbols[s],
                "raw_count": int(counts[k, s]),
                "log_enrichment": float(log_enrichment[k, s]),
            })

    pd.DataFrame(rows).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 3. CO-OCCURRENCE HELPERS
# =============================================================================

def cooccurrence_jaccard(Y: np.ndarray) -> np.ndarray:
    K = Y.shape[1]
    M = np.eye(K, dtype=float)

    for i in range(K):
        a = Y[:, i].astype(bool)

        for j in range(i + 1, K):
            b = Y[:, j].astype(bool)

            union = np.logical_or(a, b).sum()
            inter = np.logical_and(a, b).sum()

            value = (
                inter / union
                if union > 0
                else 0.0
            )

            M[i, j] = value
            M[j, i] = value

    return M


def pairwise_overlap_long(
    Y: np.ndarray,
    domain_name: str,
) -> pd.DataFrame:
    rows = []

    for i, j in combinations(range(12), 2):
        a = Y[:, i].astype(bool)
        b = Y[:, j].astype(bool)

        inter = int(np.logical_and(a, b).sum())
        union = int(np.logical_or(a, b).sum())
        jaccard = (
            inter / union
            if union > 0
            else 0.0
        )

        rows.append({
            "domain": domain_name,
            "style_id_1": STYLE_IDS[i],
            "style_1": STYLE_NAMES_EN[STYLE_IDS[i]],
            "style_id_2": STYLE_IDS[j],
            "style_2": STYLE_NAMES_EN[STYLE_IDS[j]],
            "intersection_n": inter,
            "union_n": union,
            "jaccard": float(jaccard),
        })

    out = pd.DataFrame(rows)

    out["jaccard_rank"] = (
        out["jaccard"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return out


# =============================================================================
# 4. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Final A–D-only visual genealogy analysis "
            "for Chinese Eastern Railway Paper 1."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    parser.add_argument(
        "--dino-dir",
        type=Path,
        default=DEFAULT_DINO_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=PERMUTATIONS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DIRICHLET_ALPHA,
    )
    parser.add_argument(
        "--min-support",
        type=int,
        default=MIN_DOMAIN_LINEAGE_N,
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    remove_legacy_vhi_outputs(
        args.out_dir
    )

    master_path = require(
        args.data_dir
        / "01_unique_image_multilabel_master.csv"
    )
    dino_path = require(
        args.dino_dir
        / "04_prototype_similarity_reference.csv"
    )

    print("=" * 100)
    print("PAPER 1 — FINAL VISUAL GENEALOGY")
    print("A–D ONLY (69 MORPHOLOGY SYMBOLS) | NO VHI")
    print("=" * 100)
    print(f"Master: {master_path}")
    print(f"DINO:   {dino_path}")
    print(f"Output: {args.out_dir}")
    print()

    # -------------------------------------------------------------------------
    # 4.1 Load final frozen master
    # -------------------------------------------------------------------------
    df = pd.read_csv(
        master_path,
        encoding="utf-8-sig",
    )

    missing_columns = [
        col
        for col in (
            ["dataset_role", "symbol_counts_json"]
            + [f"y_{sid}" for sid in STYLE_IDS]
        )
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Master dataset is missing required columns:\n"
            + "\n".join(missing_columns)
        )

    df = (
        df[
            df["dataset_role"].isin(
                VALID_ROLES
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    df["domain"] = df[
        "dataset_role"
    ].map(ROLE_TO_DOMAIN)

    domain_array = df[
        "domain"
    ].to_numpy()

    Y = np.vstack(
        df.apply(
            style_vector,
            axis=1,
        ).to_numpy()
    )

    X_ad = np.vstack(
        df[
            "symbol_counts_json"
        ].map(
            lambda x: count_vector(
                x,
                AD_SYMBOLS,
            )
        ).to_numpy()
    )

    # E is supplementary/context-only.
    X_e = np.vstack(
        df[
            "symbol_counts_json"
        ].map(
            lambda x: count_vector(
                x,
                E_SYMBOLS,
            )
        ).to_numpy()
    )

    df["AD_symbol_instances"] = (
        X_ad.sum(axis=1).astype(int)
    )
    df["E_context_instances"] = (
        X_e.sum(axis=1).astype(int)
    )

    labels = [
        f"{sid}_{STYLE_NAMES[sid]}"
        for sid in STYLE_IDS
    ]

    domain_counts = (
        df["domain"]
        .value_counts()
        .to_dict()
    )

    print("Frozen analytical images:")
    for domain in [
        "reference",
        "historical",
        "field",
    ]:
        print(
            f"  {domain:<10}: "
            f"{domain_counts.get(domain, 0):,}"
        )
    print(
        f"  total     : {len(df):,}"
    )
    print()

    # -------------------------------------------------------------------------
    # 4.2 A–D family summary
    # -------------------------------------------------------------------------
    family_rows = []

    for domain in [
        "reference",
        "historical",
        "field",
    ]:
        mask = (
            df["domain"]
            .eq(domain)
            .to_numpy()
        )

        Xd = X_ad[mask]

        domain_total = float(
            Xd.sum()
        )

        for family in "ABCD":
            cols = [
                i
                for i, symbol
                in enumerate(AD_SYMBOLS)
                if symbol.startswith(family)
            ]

            n_instances = int(
                Xd[:, cols].sum()
            )

            family_rows.append({
                "domain": domain,
                "family": family,
                "family_name":
                    FAMILY_NAMES[family],
                "n_symbols_in_family":
                    len(cols),
                "shape_instances":
                    n_instances,
                "AD_domain_total":
                    int(domain_total),
                "proportion_within_AD":
                    (
                        n_instances
                        / domain_total
                        if domain_total > 0
                        else np.nan
                    ),
                "included_in_main_genealogy":
                    1,
            })

    pd.DataFrame(
        family_rows
    ).to_csv(
        args.out_dir
        / "01_symbol_family_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.3 Canonical A–D reference profiles
    # -------------------------------------------------------------------------
    (
        ad_counts,
        ad_profiles,
        reference_n,
    ) = build_reference_counts_and_profiles(
        X=X_ad,
        Y=Y,
        domain=domain_array,
        symbols=AD_SYMBOLS,
        alpha=args.alpha,
    )

    export_profiles(
        path=(
            args.out_dir
            / "02_reference_symbol_profiles.csv"
        ),
        symbols=AD_SYMBOLS,
        counts=ad_counts,
        profiles=ad_profiles,
        n_images=reference_n,
        analysis_role=(
            "main_architectural_morphology_AD69"
        ),
    )

    # -------------------------------------------------------------------------
    # 4.4 A–D log enrichment and signatures
    # -------------------------------------------------------------------------
    ad_log_enrichment = (
        compute_log_enrichment(
            counts=ad_counts,
            profiles=ad_profiles,
            alpha=args.alpha,
        )
    )

    export_log_enrichment(
        path=(
            args.out_dir
            / "03_reference_symbol_log_enrichment.csv"
        ),
        symbols=AD_SYMBOLS,
        counts=ad_counts,
        log_enrichment=ad_log_enrichment,
        analysis_role=(
            "main_architectural_morphology_AD69"
        ),
    )

    export_top_signatures(
        path=(
            args.out_dir
            / "04_signature_symbols_top10.csv"
        ),
        symbols=AD_SYMBOLS,
        counts=ad_counts,
        log_enrichment=ad_log_enrichment,
        top_n=10,
        minimum_count=5,
    )

    # -------------------------------------------------------------------------
    # 4.5 Main A–D lineage similarity
    # -------------------------------------------------------------------------
    ad_similarity = cosine_similarity(
        ad_profiles
    )

    matrix_to_csv(
        ad_similarity,
        labels,
        args.out_dir
        / "05_symbol_similarity_reference.csv",
    )

    # -------------------------------------------------------------------------
    # 4.6 DINOv2 vs A–D two-layer genealogy
    # -------------------------------------------------------------------------
    dino_similarity = load_dino_similarity(
        dino_path
    )

    rho_dino_ad, p_dino_ad = (
        matrix_permutation_spearman(
            dino_similarity,
            ad_similarity,
            n_perm=args.permutations,
            seed=args.seed,
        )
    )

    convergence_df = pd.DataFrame([
        {
            "comparison":
                "DINOv2_prototype_vs_AD69_morphology",
            "spearman_upper_triangle":
                rho_dino_ad,
            "permutation_p_two_sided":
                p_dino_ad,
            "n_lineage_pairs": 66,
            "permutations":
                args.permutations,
            "main_symbol_set":
                "A-D",
            "n_main_symbols":
                len(AD_SYMBOLS),
        }
    ])

    convergence_df.to_csv(
        args.out_dir
        / "06_dinov2_symbol_similarity_convergence.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.7 Complete 66-pair two-layer genealogy table
    # -------------------------------------------------------------------------
    pair_rows = []

    for i, j in combinations(
        range(12),
        2,
    ):
        pair_rows.append({
            "pair":
                f"{STYLE_IDS[i]}-{STYLE_IDS[j]}",
            "style_id_1":
                STYLE_IDS[i],
            "style_1":
                STYLE_NAMES_EN[STYLE_IDS[i]],
            "style_id_2":
                STYLE_IDS[j],
            "style_2":
                STYLE_NAMES_EN[STYLE_IDS[j]],
            "DINOv2_similarity":
                float(
                    dino_similarity[i, j]
                ),
            "AD_vocabulary_similarity":
                float(
                    ad_similarity[i, j]
                ),
        })

    pair_df = pd.DataFrame(
        pair_rows
    )

    dino_median = float(
        pair_df[
            "DINOv2_similarity"
        ].median()
    )
    ad_median = float(
        pair_df[
            "AD_vocabulary_similarity"
        ].median()
    )

    pair_df["DINO_rank"] = (
        pair_df[
            "DINOv2_similarity"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    pair_df["Vocabulary_rank"] = (
        pair_df[
            "AD_vocabulary_similarity"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    def classify_region(row) -> str:
        syntax_high = (
            row["DINOv2_similarity"]
            >= dino_median
        )
        vocabulary_high = (
            row["AD_vocabulary_similarity"]
            >= ad_median
        )

        if (
            syntax_high
            and vocabulary_high
        ):
            return "High–High"

        if (
            (not syntax_high)
            and vocabulary_high
        ):
            return (
                "High vocabulary / Low syntax"
            )

        if (
            syntax_high
            and (not vocabulary_high)
        ):
            return (
                "Low vocabulary / High syntax"
            )

        return "Low–Low"

    pair_df[
        "relational_region"
    ] = pair_df.apply(
        classify_region,
        axis=1,
    )

    pair_df[
        "DINO_median_reference"
    ] = dino_median

    pair_df[
        "AD_median_reference"
    ] = ad_median

    pair_df.to_csv(
        args.out_dir
        / "10_pairwise_two_layer_genealogy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.8 Observed multi-lineage co-occurrence
    # -------------------------------------------------------------------------
    cooccurrence_matrices = {}
    overlap_long_by_domain = {}

    for domain in [
        "reference",
        "historical",
        "field",
    ]:
        mask = (
            df["domain"]
            .eq(domain)
            .to_numpy()
        )

        Yd = Y[mask]

        matrix = cooccurrence_jaccard(
            Yd
        )

        cooccurrence_matrices[
            domain
        ] = matrix

        matrix_to_csv(
            matrix,
            labels,
            args.out_dir
            / (
                "07_lineage_cooccurrence_"
                f"jaccard_{domain}.csv"
            ),
        )

        overlap_long_by_domain[
            domain
        ] = pairwise_overlap_long(
            Yd,
            domain,
        )

    # Complete Table-S6-style external-domain pair table.
    hist_long = (
        overlap_long_by_domain[
            "historical"
        ]
        .rename(columns={
            "intersection_n":
                "historical_intersection_n",
            "union_n":
                "historical_union_n",
            "jaccard":
                "historical_jaccard",
            "jaccard_rank":
                "historical_rank",
        })
    )

    field_long = (
        overlap_long_by_domain[
            "field"
        ]
        .rename(columns={
            "intersection_n":
                "field_intersection_n",
            "union_n":
                "field_union_n",
            "jaccard":
                "field_jaccard",
            "jaccard_rank":
                "field_rank",
        })
    )

    merge_keys = [
        "style_id_1",
        "style_1",
        "style_id_2",
        "style_2",
    ]

    external_overlap = hist_long[
        merge_keys
        + [
            "historical_intersection_n",
            "historical_union_n",
            "historical_jaccard",
            "historical_rank",
        ]
    ].merge(
        field_long[
            merge_keys
            + [
                "field_intersection_n",
                "field_union_n",
                "field_jaccard",
                "field_rank",
            ]
        ],
        on=merge_keys,
        how="outer",
        validate="one_to_one",
    )

    external_overlap.to_csv(
        args.out_dir
        / "11_pairwise_cooccurrence_complete.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.9 Matrix comparisons:
    #     DINO vs co-occurrence
    #     A–D vs co-occurrence
    # -------------------------------------------------------------------------
    matrix_comparison_rows = []

    for domain in [
        "reference",
        "historical",
        "field",
    ]:
        cooc = (
            cooccurrence_matrices[
                domain
            ]
        )

        for comparison_name, matrix in [
            (
                "DINOv2_vs_cooccurrence",
                dino_similarity,
            ),
            (
                "AD69_vs_cooccurrence",
                ad_similarity,
            ),
        ]:
            rho, p = (
                matrix_permutation_spearman(
                    matrix,
                    cooc,
                    n_perm=args.permutations,
                    seed=(
                        args.seed
                        + len(
                            matrix_comparison_rows
                        )
                        + 1
                    ),
                )
            )

            matrix_comparison_rows.append({
                "domain": domain,
                "comparison":
                    comparison_name,
                "spearman_upper_triangle":
                    rho,
                "permutation_p_two_sided":
                    p,
                "n_lineage_pairs": 66,
                "permutations":
                    args.permutations,
            })

    pd.DataFrame(
        matrix_comparison_rows
    ).to_csv(
        args.out_dir
        / "08_similarity_convergence_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.10 Cross-domain A–D continuity
    #       continuity = 1 - Jensen-Shannon distance
    # -------------------------------------------------------------------------
    continuity_rows = []

    for target_domain in [
        "historical",
        "field",
    ]:
        target_domain_mask = (
            df["domain"]
            .eq(target_domain)
            .to_numpy()
        )

        for k, sid in enumerate(
            STYLE_IDS
        ):
            target_mask = (
                target_domain_mask
                & (Y[:, k] == 1)
            )

            n_target = int(
                target_mask.sum()
            )

            if n_target == 0:
                continue

            target_counts = (
                X_ad[
                    target_mask
                ]
                .sum(axis=0)
            )

            target_profile = (
                smoothed_profile(
                    target_counts,
                    alpha=args.alpha,
                )
            )

            # scipy jensenshannon with base=2 gives a distance in [0,1].
            js_distance = float(
                jensenshannon(
                    ad_profiles[k],
                    target_profile,
                    base=2.0,
                )
            )

            continuity = (
                1.0 - js_distance
            )

            continuity_rows.append({
                "style_id": sid,
                "style_name":
                    STYLE_NAMES[sid],
                "style_name_en":
                    STYLE_NAMES_EN[sid],
                "target_domain":
                    target_domain,
                "n_reference_single_lineage":
                    int(reference_n[k]),
                "n_target_positive":
                    n_target,
                "adequate_target_support":
                    int(
                        n_target
                        >= args.min_support
                    ),
                "minimum_support_rule":
                    args.min_support,
                "symbol_set":
                    "A-D",
                "n_symbols":
                    len(AD_SYMBOLS),
                "jensen_shannon_distance":
                    js_distance,
                "symbol_continuity_index":
                    continuity,
            })

    continuity_df = pd.DataFrame(
        continuity_rows
    )

    continuity_df.to_csv(
        args.out_dir
        / "09_symbol_continuity_by_lineage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # 4.11 Supplementary E context only
    #       E NEVER enters main genealogy / continuity.
    # -------------------------------------------------------------------------
    (
        e_counts,
        e_profiles,
        e_reference_n,
    ) = build_reference_counts_and_profiles(
        X=X_e,
        Y=Y,
        domain=domain_array,
        symbols=E_SYMBOLS,
        alpha=args.alpha,
    )

    e_log_enrichment = (
        compute_log_enrichment(
            counts=e_counts,
            profiles=e_profiles,
            alpha=args.alpha,
        )
    )

    export_log_enrichment(
        path=(
            args.out_dir
            / "12_E_context_reference_log_enrichment.csv"
        ),
        symbols=E_SYMBOLS,
        counts=e_counts,
        log_enrichment=e_log_enrichment,
        analysis_role=(
            "supplementary_context_only"
        ),
    )

    export_profiles(
        path=(
            args.out_dir
            / "13_E_context_reference_profiles.csv"
        ),
        symbols=E_SYMBOLS,
        counts=e_counts,
        profiles=e_profiles,
        n_images=e_reference_n,
        analysis_role=(
            "supplementary_context_only"
        ),
    )

    # -------------------------------------------------------------------------
    # 4.12 Summary
    # -------------------------------------------------------------------------
    adequate_hist = continuity_df[
        (
            continuity_df[
                "target_domain"
            ] == "historical"
        )
        & (
            continuity_df[
                "adequate_target_support"
            ] == 1
        )
    ]

    adequate_field = continuity_df[
        (
            continuity_df[
                "target_domain"
            ] == "field"
        )
        & (
            continuity_df[
                "adequate_target_support"
            ] == 1
        )
    ]

    region_counts = (
        pair_df[
            "relational_region"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "analysis_version":
            "FINAL_AD69_NO_VHI",
        "seed":
            args.seed,
        "permutations":
            args.permutations,
        "dirichlet_alpha":
            args.alpha,
        "n_images_analyzed":
            int(len(df)),
        "domain_image_counts": {
            key: int(value)
            for key, value
            in domain_counts.items()
        },
        "canonical_reference_rule":
            "reference + single-lineage only",
        "n_visual_lineages":
            12,
        "main_symbol_families":
            ["A", "B", "C", "D"],
        "n_main_AD_symbols":
            len(AD_SYMBOLS),
        "E_context_symbols":
            len(E_SYMBOLS),
        "E_context_role":
            (
                "supplementary only; excluded "
                "from genealogy and continuity"
            ),
        "VHI_status":
            "REMOVED",
        "minimum_target_support_for_main_continuity_interpretation":
            args.min_support,
        "DINOv2_vs_AD_similarity_spearman":
            rho_dino_ad,
        "DINOv2_vs_AD_similarity_permutation_p":
            p_dino_ad,
        "DINO_similarity_median":
            dino_median,
        "AD_similarity_median":
            ad_median,
        "two_layer_region_counts":
            {
                str(k): int(v)
                for k, v
                in region_counts.items()
            },
        "mean_AD_continuity_historical_adequate_support":
            (
                float(
                    adequate_hist[
                        "symbol_continuity_index"
                    ].mean()
                )
                if len(adequate_hist)
                else None
            ),
        "mean_AD_continuity_field_adequate_support":
            (
                float(
                    adequate_field[
                        "symbol_continuity_index"
                    ].mean()
                )
                if len(adequate_field)
                else None
            ),
        "output_notes": {
            "02_to_06":
                "A-D main morphology only",
            "07_and_11":
                (
                    "Observed multi-label "
                    "co-occurrence"
                ),
            "09":
                (
                    "A-D continuity = "
                    "1 - Jensen-Shannon distance"
                ),
            "12_to_13":
                (
                    "E context supplementary only"
                ),
        },
    }

    (
        args.out_dir
        / "00_stage4_summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 4.13 Console QA
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("FINAL VISUAL GENEALOGY COMPLETE")
    print("=" * 100)
    print(
        f"Images analyzed: {len(df):,}"
    )
    print(
        f"Main vocabulary: A–D = "
        f"{len(AD_SYMBOLS)} symbols"
    )
    print(
        f"E context: {len(E_SYMBOLS)} symbols "
        f"(supplementary only)"
    )
    print("VHI: REMOVED")
    print()
    print(
        f"DINOv2 vs A–D similarity: "
        f"rho = {rho_dino_ad:.6f}, "
        f"p = {p_dino_ad:.6f}"
    )
    print(
        f"Median DINO similarity: "
        f"{dino_median:.6f}"
    )
    print(
        f"Median A–D similarity: "
        f"{ad_median:.6f}"
    )
    print()
    print("Relational-region counts:")
    for key in [
        "High–High",
        "High vocabulary / Low syntax",
        "Low vocabulary / High syntax",
        "Low–Low",
    ]:
        print(
            f"  {key:<31}: "
            f"{region_counts.get(key, 0)}"
        )
    print()
    print(
        "Mean A–D continuity, adequate support only:"
    )
    print(
        "  Historical: "
        f"{summary['mean_AD_continuity_historical_adequate_support']}"
    )
    print(
        "  Field:      "
        f"{summary['mean_AD_continuity_field_adequate_support']}"
    )
    print()
    print("Key final outputs:")
    for name in [
        "04_signature_symbols_top10.csv",
        "05_symbol_similarity_reference.csv",
        "06_dinov2_symbol_similarity_convergence.csv",
        "08_similarity_convergence_all.csv",
        "09_symbol_continuity_by_lineage.csv",
        "10_pairwise_two_layer_genealogy.csv",
        "11_pairwise_cooccurrence_complete.csv",
        "12_E_context_reference_log_enrichment.csv",
    ]:
        print(f"  - {name}")

    print()
    print(f"Output: {args.out_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
