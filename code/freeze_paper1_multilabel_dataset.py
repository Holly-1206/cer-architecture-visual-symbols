# -*- coding: utf-8 -*-
"""
freeze_paper1_multilabel_dataset.py
Paper 1 — Chinese Eastern Railway visual genealogies
Stage 3: deduplication, multi-lineage target construction, symbol-ontology merge,
quality-control flags, and five-fold development split.

INPUT
-----
<PROJECT_ROOT>/prepared_corpus_v2/01_master_manifest.csv

This file is produced by prepare_labelme_corpus_v2.py.

CORE DESIGN
-----------
1) One normalized pixel hash = one unique image.
2) Exact duplicate images in several style folders are NOT counted several times.
3) Their style folders are combined into a 12-dimensional multi-label target.
4) Symbol annotations are mapped to the final 84-code A–E ontology.
5) For duplicate JSON copies of one image, the JSON containing the largest number
   of valid core symbol instances is selected as the annotation master.
6) Obvious non-building / "not applicable" records are flagged rather than deleted.
7) Only the expanded reference corpus is assigned 5 development folds.
   Historical archive and field photographs remain external domains.

NO ORIGINAL FILES ARE MODIFIED.
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# 0. PATHS
# =============================================================================

PREP_DIR = Path(str(PROJECT_ROOT / "prepared_corpus_v2"))
INPUT_MANIFEST = PREP_DIR / "01_master_manifest.csv"

OUT_DIR = Path(str(PROJECT_ROOT / "paper1_frozen_dataset"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260916
N_FOLDS = 5

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
STYLE_IDS = list(STYLE_NAMES.keys())

# =============================================================================
# 1. FINAL 84-CODE VISUAL SYMBOL ONTOLOGY
# =============================================================================

SEMANTIC_TO_AE = {
    # A — roof/top silhouette
    "ROOF_ONION_DOME": "A1",
    "ROOF_TENT": "A2",
    "ROOF_GABLE": "A3",
    "ROOF_HIPPED": "A4",
    "ROOF_MULTI_SLOPE": "A5",
    "ROOF_METAL_SHEET": "A6",
    "ROOF_DORMER": "A7",
    "ROOF_FLAT": "A8",
    "ROOF_PARAPET": "A9",
    "ROOF_CROSS": "A10",
    "ROOF_CHIMNEY": "A11",
    "ROOF_EAVE_OVERHANG": "A12",

    # B — openings
    "WIN_SEMICIRCULAR_ARCH": "B1",
    "WIN_SEGMENTAL_ARCH": "B2",
    "WIN_POINTED_ARCH": "B3",
    "WIN_RECTANGULAR": "B4",
    "WIN_PAIRED": "B5",
    "WIN_TRIPLE": "B6",
    "WIN_BAY": "B7",
    "WIN_SMALL_VENT": "B8",
    "WIN_GRID": "B9",
    "WIN_SURROUND": "B10",
    "WIN_KEYSTONE": "B11",
    "WIN_SILL": "B12",
    "DOOR_ARCH": "B13",
    "DOOR_PORCH": "B14",
    "DOOR_STEPS": "B15",
    "DOOR_FANLIGHT": "B16",

    # C — wall/material/structure
    "MAT_RED_BRICK": "C1",
    "MAT_YELLOW_BRICK": "C2",
    "MAT_BRICK_PAINT": "C3",
    "MAT_CEMENT_PLASTER": "C4",
    "MAT_STONE_BASE": "C5",
    "MAT_STONE_VENEER": "C6",
    "MAT_TIMBER_LOG": "C7",
    "MAT_VERTICAL_WOOD": "C8",
    "MAT_RUSTICATION": "C9",
    "MAT_QUOIN": "C10",
    "STRUCT_BRICK_WOOD": "C11",
    "STRUCT_WOOD": "C12",
    "STRUCT_BRICK_STONE": "C13",
    "STRUCT_BRICK_CONCRETE": "C14",
    "STRUCT_STEEL_TRUSS": "C15",
    "STRUCT_CONCRETE_ARCH": "C16",
    "STRUCT_WAVE_ARCH": "C17",
    "WALL_BRICK_BOND": "C18",
    "WALL_INTERLOCK": "C19",
    "WALL_PILASTER": "C20",

    # D — façade decoration / motifs
    "DEC_CORNICE": "D1",
    "DEC_WHITE_BAND": "D2",
    "DEC_PEDIMENT_TRI": "D3",
    "DEC_PEDIMENT_CURVE": "D4",
    "DEC_VOLUTE": "D5",
    "DEC_FLORAL_RELIEF": "D6",
    "DEC_ART_NOUVEAU_CURVE": "D7",
    "DEC_RUS_WOOD_LACE": "D8",
    "DEC_COLOR_TRIM": "D9",
    "DEC_CHN_CLOUD": "D10",
    "DEC_CHN_HUI": "D11",
    "DEC_CHN_BAT": "D12",
    "DEC_CHN_POMEGRANATE": "D13",
    "DEC_CHN_GRAPE": "D14",
    "DEC_CHN_PEONY": "D15",
    "DEC_CHN_KNOT": "D16",
    "DEC_CHN_COIN": "D17",
    "DEC_DOUGONG": "D18",
    "DEC_BALUSTRADE": "D19",
    "DEC_IRONWORK": "D20",
    "DEC_PLAQUE": "D21",

    # E — railway / infrastructure
    "RAIL_STATION_CANOPY": "E1",
    "RAIL_PLATFORM": "E2",
    "RAIL_TRACK": "E3",
    "RAIL_ROUNDHOUSE": "E4",
    "RAIL_TURNTABLE": "E5",
    "RAIL_ENGINE_SHED_DOOR": "E6",
    "RAIL_WATER_TOWER": "E7",
    "RAIL_WAREHOUSE_DOOR": "E8",
    "RAIL_CHIMNEY": "E9",
    "RAIL_BRIDGE_TRUSS": "E10",
    "RAIL_STONE_ARCH_BRIDGE": "E11",
    "RAIL_TUNNEL_PORTAL": "E12",
    "RAIL_CULVERT": "E13",
    "RAIL_SIGNAL": "E14",
    "RAIL_FENCE": "E15",
}

VALID_AE = set(SEMANTIC_TO_AE.values())

# Context annotations that are useful for auditing but are not core architectural symbols.
AUXILIARY_PREFIXES = ("AUX_", "SPACE_")

# This named label is architecturally meaningful but does not have an exact one-to-one
# code in the supplied codebook. Keep for manual review rather than silently merging.
MANUAL_SYMBOL_REVIEW = {"DEC_RUSTICATED_BAND"}

# Explicit malformed / out-of-ontology labels observed in the audit.
KNOWN_REVIEW_LABELS = {
    "2", "F1", "V10", "AD1", "吃",
    "A15", "A19", "B19", "C21",
}

NONBUILDING_TOKENS = (
    "非建筑",
    "NON_BUILDING",
    "不适用",
    "未归入给定风格",
)

# =============================================================================
# 2. UTILITIES
# =============================================================================

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def normalize_label(x: Any) -> str:
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x))
    s = re.sub(r"\s+", " ", s).strip().rstrip("\\/").strip()

    # Safe format normalization only.
    m = re.fullmatch(r"([A-Ea-e])(\d+)", s)
    if m:
        s = f"{m.group(1).upper()}{m.group(2)}"
    return s


def map_symbol(raw: Any) -> Tuple[str | None, str]:
    """
    Returns (A-E code or None, status)
    status:
      core
      auxiliary
      manual_review
      invalid
    """
    s = normalize_label(raw)

    if s in VALID_AE:
        return s, "core"

    if s in SEMANTIC_TO_AE:
        return SEMANTIC_TO_AE[s], "core"

    if s.startswith(AUXILIARY_PREFIXES):
        return None, "auxiliary"

    if s in MANUAL_SYMBOL_REVIEW:
        return None, "manual_review"

    if s in KNOWN_REVIEW_LABELS:
        return None, "manual_review"

    # Any A-E-looking code outside the supplied 84-code ontology.
    if re.fullmatch(r"[A-E]\d+", s):
        return None, "manual_review"

    return None, "invalid"


def load_json(path: Path) -> Dict[str, Any] | None:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=enc) as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
    return None


def parse_symbol_counts(json_path: Path) -> Tuple[Counter, Counter, int]:
    """
    Returns:
      core_counts      Counter(A1...E15)
      noncore_counts   Counter(raw non-core labels)
      valid_core_n     total number of mapped core shape instances
    """
    data = load_json(json_path)
    if not data:
        return Counter(), Counter({"__JSON_READ_ERROR__": 1}), 0

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        return Counter(), Counter({"__INVALID_SHAPES__": 1}), 0

    core = Counter()
    other = Counter()

    for sh in shapes:
        if not isinstance(sh, dict):
            continue
        raw = sh.get("label", "")
        mapped, status = map_symbol(raw)
        if mapped:
            core[mapped] += 1
        else:
            other[f"{status}:{normalize_label(raw)}"] += 1

    return core, other, sum(core.values())


def serialize_counts(c: Counter) -> str:
    return json.dumps(dict(sorted(c.items())), ensure_ascii=False, separators=(",", ":"))


def style_vector(style_set: set[str]) -> Dict[str, int]:
    return {f"y_{sid}": int(sid in style_set) for sid in STYLE_IDS}


def filename_qc(json_paths: List[str]) -> Tuple[int, str]:
    hits = []
    text = " || ".join(json_paths)
    low = text.lower()
    for tok in NONBUILDING_TOKENS:
        if tok.lower() in low:
            hits.append(tok)
    return int(bool(hits)), "|".join(sorted(set(hits)))


# =============================================================================
# 3. DETERMINISTIC GREEDY MULTI-LABEL FOLD ASSIGNMENT
# =============================================================================

def assign_multilabel_folds(rows: List[Dict[str, Any]], n_folds: int, seed: int) -> None:
    """
    Greedy label-balancing split with no third-party package.

    Samples carrying rarer labels and/or several labels are allocated first.
    Objective balances:
      - each label count per fold
      - overall fold size
    """
    rng = random.Random(seed)

    label_totals = Counter()
    for r in rows:
        for sid in STYLE_IDS:
            if int(r[f"y_{sid}"]) == 1:
                label_totals[sid] += 1

    target_label = {sid: label_totals[sid] / n_folds for sid in STYLE_IDS}
    target_size = len(rows) / n_folds

    def rarity_score(r):
        labs = [sid for sid in STYLE_IDS if int(r[f"y_{sid}"]) == 1]
        # higher = allocate earlier
        rarity = sum(1.0 / max(label_totals[sid], 1) for sid in labs)
        return (len(labs), rarity, rng.random())

    ordered = sorted(rows, key=rarity_score, reverse=True)

    fold_label = [Counter() for _ in range(n_folds)]
    fold_size = [0] * n_folds

    for r in ordered:
        labs = [sid for sid in STYLE_IDS if int(r[f"y_{sid}"]) == 1]

        best_fold = None
        best_score = None

        for f in range(n_folds):
            score = 0.0

            # label-balance penalty after hypothetical assignment
            for sid in labs:
                after = fold_label[f][sid] + 1
                target = max(target_label[sid], 1e-6)
                score += ((after - target) / target) ** 2

            # fold-size penalty
            after_size = fold_size[f] + 1
            score += 0.20 * ((after_size - target_size) / max(target_size, 1.0)) ** 2

            # tiny deterministic jitter to break ties
            score += rng.random() * 1e-9

            if best_score is None or score < best_score:
                best_score = score
                best_fold = f

        r["fold"] = int(best_fold)
        fold_size[best_fold] += 1
        for sid in labs:
            fold_label[best_fold][sid] += 1


# =============================================================================
# 4. MAIN
# =============================================================================

def main() -> None:
    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {INPUT_MANIFEST}")

    raw = read_csv(INPUT_MANIFEST)

    # Only recovered images can enter frozen image datasets.
    recovered = [r for r in raw if r.get("pixel_sha256") and r.get("extracted_image_path")]

    # Group the same visual image within a source.
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in recovered:
        groups[(r["source"], r["pixel_sha256"])].append(r)

    frozen: List[Dict[str, Any]] = []
    qc_rows: List[Dict[str, Any]] = []
    symbol_review_counter = Counter()

    for (source, psha), recs in sorted(groups.items()):
        style_set = {r["style_id"] for r in recs if r["style_id"] in STYLE_IDS}

        parsed = []
        for r in recs:
            jp = Path(r["json_path"])
            core, other, core_n = parse_symbol_counts(jp)
            parsed.append((core_n, r, core, other))

            for k, v in other.items():
                symbol_review_counter[k] += v

        # The most completely annotated exact copy is the annotation master.
        parsed.sort(key=lambda x: (x[0], int(float(x[1].get("shape_count") or 0))), reverse=True)
        core_n, master, core_counts, noncore_counts = parsed[0]

        all_json_paths = [r["json_path"] for r in recs]
        nonbuilding_flag, nonbuilding_reason = filename_qc(all_json_paths)

        row: Dict[str, Any] = {
            "source": source,
            "pixel_sha256": psha,
            "dhash": master.get("dhash", ""),
            "image_path": master["extracted_image_path"],
            "width": master.get("width", ""),
            "height": master.get("height", ""),
            "n_original_records": len(recs),
            "n_style_labels": len(style_set),
            "style_ids": "|".join(sorted(style_set)),
            "style_names": "|".join(STYLE_NAMES[s] for s in sorted(style_set)),
            "multi_lineage": int(len(style_set) > 1),
            "annotation_master_json": master["json_path"],
            "all_json_paths": " || ".join(all_json_paths),
            "core_symbol_instances": core_n,
            "core_symbol_types": len(core_counts),
            "symbol_counts_json": serialize_counts(core_counts),
            "symbol_presence": "|".join(sorted(core_counts, key=lambda x: (x[0], int(x[1:])))),
            "noncore_symbol_counts_json": serialize_counts(noncore_counts),
            "nonbuilding_flag": nonbuilding_flag,
            "nonbuilding_reason": nonbuilding_reason,
            # Style model uses image evidence; empty symbol annotation is not an automatic exclusion.
            "eligible_style_model": int(not nonbuilding_flag and len(style_set) > 0),
        }
        row.update(style_vector(style_set))
        frozen.append(row)

        if len(style_set) > 1 or nonbuilding_flag or noncore_counts:
            qc_rows.append({
                "source": source,
                "pixel_sha256": psha,
                "image_path": master["extracted_image_path"],
                "style_ids": row["style_ids"],
                "style_names": row["style_names"],
                "multi_lineage": row["multi_lineage"],
                "nonbuilding_flag": nonbuilding_flag,
                "nonbuilding_reason": nonbuilding_reason,
                "noncore_symbol_counts_json": row["noncore_symbol_counts_json"],
                "all_json_paths": row["all_json_paths"],
            })

    # -------------------------------------------------------------------------
    # Development / external-domain assignment
    # -------------------------------------------------------------------------
    reference = [
        r for r in frozen
        if r["source"] == "expanded_reference" and int(r["eligible_style_model"]) == 1
    ]
    historical = [
        r for r in frozen
        if r["source"] == "historical_album" and int(r["eligible_style_model"]) == 1
    ]
    field = [
        r for r in frozen
        if r["source"] == "field_photographs" and int(r["eligible_style_model"]) == 1
    ]

    assign_multilabel_folds(reference, N_FOLDS, RANDOM_SEED)

    for r in frozen:
        r["dataset_role"] = ""
        r["fold"] = r.get("fold", "")

    ref_lookup = {r["pixel_sha256"]: r for r in reference}
    for r in frozen:
        if r["source"] == "expanded_reference" and r["pixel_sha256"] in ref_lookup:
            r["dataset_role"] = "development_reference"
            r["fold"] = ref_lookup[r["pixel_sha256"]]["fold"]
        elif r["source"] == "historical_album" and int(r["eligible_style_model"]) == 1:
            r["dataset_role"] = "external_historical"
        elif r["source"] == "field_photographs" and int(r["eligible_style_model"]) == 1:
            r["dataset_role"] = "external_field"
        else:
            r["dataset_role"] = "qc_excluded_from_style_model"

    # -------------------------------------------------------------------------
    # Style combination summary
    # -------------------------------------------------------------------------
    combo_counter = Counter((r["source"], r["style_ids"]) for r in frozen)
    combo_rows = []
    for (source, combo), n in sorted(combo_counter.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        combo_rows.append({
            "source": source,
            "style_ids": combo,
            "style_names": "|".join(STYLE_NAMES[s] for s in combo.split("|") if s in STYLE_NAMES),
            "n_unique_images": n,
            "n_labels": len(combo.split("|")) if combo else 0,
        })

    # -------------------------------------------------------------------------
    # Symbol frequencies by source and lineage
    # For multi-lineage images, the same image contributes its symbol profile
    # to every lineage membership. This is intentional for lineage profiling.
    # -------------------------------------------------------------------------
    sym_counter = Counter()
    image_counter = Counter()

    for r in frozen:
        if not r["symbol_counts_json"]:
            continue
        counts = json.loads(r["symbol_counts_json"])
        styles = r["style_ids"].split("|") if r["style_ids"] else []

        for sid in styles:
            image_counter[(r["source"], sid)] += 1
            for sym, n in counts.items():
                sym_counter[(r["source"], sid, sym)] += int(n)

    symbol_rows = []
    for (source, sid, sym), n in sorted(sym_counter.items()):
        symbol_rows.append({
            "source": source,
            "style_id": sid,
            "style_name": STYLE_NAMES[sid],
            "symbol_code": sym,
            "shape_instances": n,
            "unique_images_in_lineage": image_counter[(source, sid)],
        })

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    base_fields = [
        "source", "pixel_sha256", "dhash", "image_path", "width", "height",
        "n_original_records", "n_style_labels", "style_ids", "style_names",
        "multi_lineage", "annotation_master_json", "all_json_paths",
        "core_symbol_instances", "core_symbol_types", "symbol_counts_json",
        "symbol_presence", "noncore_symbol_counts_json",
        "nonbuilding_flag", "nonbuilding_reason",
        "eligible_style_model", "dataset_role", "fold",
    ] + [f"y_{sid}" for sid in STYLE_IDS]

    write_csv(OUT_DIR / "01_unique_image_multilabel_master.csv", frozen, base_fields)
    write_csv(OUT_DIR / "02_reference_development_5fold.csv", reference, base_fields)
    write_csv(OUT_DIR / "03_historical_external.csv", historical, base_fields)
    write_csv(OUT_DIR / "04_field_external.csv", field, base_fields)
    write_csv(
        OUT_DIR / "05_symbol_frequency_by_source_lineage.csv",
        symbol_rows,
        ["source", "style_id", "style_name", "symbol_code", "shape_instances", "unique_images_in_lineage"],
    )
    write_csv(
        OUT_DIR / "06_lineage_combinations.csv",
        combo_rows,
        ["source", "style_ids", "style_names", "n_unique_images", "n_labels"],
    )
    write_csv(
        OUT_DIR / "07_qc_review_images.csv",
        qc_rows,
        [
            "source", "pixel_sha256", "image_path", "style_ids", "style_names",
            "multi_lineage", "nonbuilding_flag", "nonbuilding_reason",
            "noncore_symbol_counts_json", "all_json_paths",
        ],
    )

    symbol_review_rows = [
        {"noncore_label": k, "shape_instances": v}
        for k, v in symbol_review_counter.most_common()
    ]
    write_csv(
        OUT_DIR / "08_noncore_symbol_review.csv",
        symbol_review_rows,
        ["noncore_label", "shape_instances"],
    )

    # -------------------------------------------------------------------------
    # Human-readable freeze summary
    # -------------------------------------------------------------------------
    by_source = defaultdict(list)
    for r in frozen:
        by_source[r["source"]].append(r)

    md = []
    md.append("# Paper 1 Frozen Dataset Summary")
    md.append("")
    md.append("## Unique recovered images")
    md.append("")
    md.append("| Domain | Unique images | Multi-lineage | Eligible style-model images |")
    md.append("|---|---:|---:|---:|")
    for src in ("expanded_reference", "historical_album", "field_photographs"):
        rr = by_source[src]
        md.append(
            f"| {src} | {len(rr)} | "
            f"{sum(int(x['multi_lineage']) for x in rr)} | "
            f"{sum(int(x['eligible_style_model']) for x in rr)} |"
        )
    md.append("")
    md.append("## Frozen modeling roles")
    md.append("")
    md.append(f"- Development/reference: {len(reference)} unique images, 5 folds.")
    md.append(f"- Historical external domain: {len(historical)} unique images.")
    md.append(f"- Field external domain: {len(field)} unique images.")
    md.append("")
    md.append("## Multi-label principle")
    md.append("")
    md.append(
        "Exact duplicate pixels found under several lineage folders are represented once, "
        "with the union of those lineage labels. This avoids both train/test leakage and "
        "the conceptual error of forcing visually hybrid cases into one mutually exclusive class."
    )
    md.append("")
    md.append("## Symbol ontology")
    md.append("")
    md.append(
        "Only labels mapping to the supplied 84-code A–E ontology are used in core symbol profiles. "
        "Auxiliary/context labels and malformed/out-of-ontology labels remain in review outputs."
    )
    md.append("")

    (OUT_DIR / "00_frozen_dataset_summary.md").write_text("\n".join(md), encoding="utf-8")

    print("=" * 100)
    print("PAPER 1 DATASET FREEZE COMPLETE")
    print("=" * 100)
    print(f"Unique recovered images:       {len(frozen):,}")
    print(f"Development/reference:         {len(reference):,}")
    print(f"Historical external:           {len(historical):,}")
    print(f"Field external:                {len(field):,}")
    print(f"Multi-lineage unique images:   {sum(int(r['multi_lineage']) for r in frozen):,}")
    print(f"QC review rows:                {len(qc_rows):,}")
    print(f"Output: {OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
