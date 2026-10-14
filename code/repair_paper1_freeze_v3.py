# -*- coding: utf-8 -*-
"""
repair_paper1_freeze_v3.py

Final pre-training repair/QC for Paper 1.

Fixes two issues discovered after auditing the frozen master:
1) Rebuilds a genuinely balanced 5-fold multi-label split.
2) Quarantines reference images whose filenames explicitly contain
   "非目标" or "硬负样本" so they are not treated as positive lineage examples.

No original files are modified.

INPUT
-----
<PROJECT_ROOT>/paper1_frozen_dataset/
    01_unique_image_multilabel_master.csv

OUTPUT
------
<PROJECT_ROOT>/paper1_frozen_dataset_v3/
    00_frozen_dataset_summary_v3.md
    01_unique_image_multilabel_master.csv
    02_reference_development_5fold.csv
    03_historical_external.csv
    04_field_external.csv
    05_reference_quarantine.csv
    09_fold_balance.csv
    10_domain_lineage_prevalence.csv
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

IN_DIR = Path(str(PROJECT_ROOT / "paper1_frozen_dataset"))
OUT_DIR = Path(str(PROJECT_ROOT / "paper1_frozen_dataset_v3"))
IN_MASTER = IN_DIR / "01_unique_image_multilabel_master.csv"

SEED = 20260916
N_FOLDS = 5

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

QUARANTINE_TOKENS = ("非目标", "硬负样本")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def intv(x: Any) -> int:
    return int(float(x))


def has_quarantine_token(row: Dict[str, str]) -> tuple[int, str]:
    text = row.get("all_json_paths", "") or ""
    hits = [t for t in QUARANTINE_TOKENS if t in text]
    return int(bool(hits)), "|".join(hits)


def assign_balanced_multilabel_folds(
    rows: List[Dict[str, Any]],
    n_folds: int = 5,
    seed: int = 20260916,
) -> None:
    rng = random.Random(seed)

    totals = {
        sid: sum(intv(r[f"y_{sid}"]) for r in rows)
        for sid in STYLE_IDS
    }
    targets = {sid: totals[sid] / n_folds for sid in STYLE_IDS}
    target_size = len(rows) / n_folds
    max_size = math.ceil(target_size)

    def order_key(r):
        labs = [sid for sid in STYLE_IDS if intv(r[f"y_{sid}"]) == 1]
        rarity = sum(1.0 / max(totals[s], 1) for s in labs)
        return (rarity, len(labs), rng.random())

    ordered = sorted(rows, key=order_key, reverse=True)

    fold_counts = [Counter() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for r in ordered:
        labs = [sid for sid in STYLE_IDS if intv(r[f"y_{sid}"]) == 1]

        best_fold = None
        best_score = None

        for f in range(n_folds):
            label_need = 0.0
            for sid in labs:
                target = max(targets[sid], 1e-12)
                deficit = max(target - fold_counts[f][sid], 0.0)
                label_need += deficit / target

            size_need = max(target_size - fold_sizes[f], 0.0) / max(target_size, 1.0)
            overflow = max((fold_sizes[f] + 1) - max_size, 0)

            score = label_need + 0.50 * size_need - 10.0 * overflow + rng.random() * 1e-10

            if best_score is None or score > best_score:
                best_score = score
                best_fold = f

        r["fold"] = str(best_fold)
        fold_sizes[best_fold] += 1
        for sid in labs:
            fold_counts[best_fold][sid] += 1


def main():
    if not IN_MASTER.exists():
        raise FileNotFoundError(IN_MASTER)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(IN_MASTER)
    fields = list(rows[0].keys())

    # Add audit columns without changing any original labels.
    extra_fields = ["reference_quarantine_flag", "reference_quarantine_reason"]
    for ef in extra_fields:
        if ef not in fields:
            fields.append(ef)

    # Exact-image integrity.
    keys = [(r["source"], r["pixel_sha256"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate (source, pixel_sha256) rows remain.")

    hash_sources = defaultdict(set)
    for r in rows:
        hash_sources[r["pixel_sha256"]].add(r["source"])
    cross_domain = [h for h, ss in hash_sources.items() if len(ss) > 1]
    if cross_domain:
        raise RuntimeError(
            f"Cross-domain exact duplicate hashes found: {len(cross_domain)}"
        )

    quarantine = []

    for r in rows:
        qflag, reason = (0, "")
        if r["source"] == "expanded_reference":
            qflag, reason = has_quarantine_token(r)

        r["reference_quarantine_flag"] = str(qflag)
        r["reference_quarantine_reason"] = reason

        if qflag:
            quarantine.append(r)

    reference = [
        r for r in rows
        if r["source"] == "expanded_reference"
        and intv(r["eligible_style_model"]) == 1
        and intv(r["reference_quarantine_flag"]) == 0
    ]

    historical = [
        r for r in rows
        if r["source"] == "historical_album"
        and intv(r["eligible_style_model"]) == 1
    ]

    field = [
        r for r in rows
        if r["source"] == "field_photographs"
        and intv(r["eligible_style_model"]) == 1
    ]

    assign_balanced_multilabel_folds(reference, N_FOLDS, SEED)
    fold_lookup = {r["pixel_sha256"]: r["fold"] for r in reference}

    for r in rows:
        if (
            r["source"] == "expanded_reference"
            and intv(r["eligible_style_model"]) == 1
            and intv(r["reference_quarantine_flag"]) == 0
        ):
            r["dataset_role"] = "development_reference"
            r["fold"] = fold_lookup[r["pixel_sha256"]]

        elif (
            r["source"] == "expanded_reference"
            and intv(r["reference_quarantine_flag"]) == 1
        ):
            r["dataset_role"] = "qc_reference_quarantine"
            r["fold"] = ""

        elif (
            r["source"] == "historical_album"
            and intv(r["eligible_style_model"]) == 1
        ):
            r["dataset_role"] = "external_historical"
            r["fold"] = ""

        elif (
            r["source"] == "field_photographs"
            and intv(r["eligible_style_model"]) == 1
        ):
            r["dataset_role"] = "external_field"
            r["fold"] = ""

        else:
            r["dataset_role"] = "qc_excluded_from_style_model"
            r["fold"] = ""

    # Fold balance.
    balance_rows = []
    for f in range(N_FOLDS):
        fr = [r for r in reference if int(r["fold"]) == f]
        br = {
            "fold": f,
            "n_images": len(fr),
            "n_multilineage": sum(intv(r["multi_lineage"]) for r in fr),
        }
        for sid in STYLE_IDS:
            br[f"n_{sid}"] = sum(intv(r[f"y_{sid}"]) for r in fr)
        balance_rows.append(br)

    fold_sizes = [r["n_images"] for r in balance_rows]

    if any(x == 0 for x in fold_sizes):
        raise RuntimeError(f"Empty fold: {fold_sizes}")

    if max(fold_sizes) - min(fold_sizes) > 1:
        raise RuntimeError(f"Fold size imbalance: {fold_sizes}")

    for sid in STYLE_IDS:
        vals = [r[f"n_{sid}"] for r in balance_rows]
        if max(vals) - min(vals) > 2:
            raise RuntimeError(f"Poor balance for lineage {sid}: {vals}")

    # Domain prevalence.
    prevalence_rows = []
    for domain, data in [
        ("development_reference", reference),
        ("external_historical", historical),
        ("external_field", field),
    ]:
        for sid in STYLE_IDS:
            npos = sum(intv(r[f"y_{sid}"]) for r in data)
            prevalence_rows.append({
                "domain": domain,
                "style_id": sid,
                "style_name": STYLE_NAMES[sid],
                "n_domain_images": len(data),
                "n_positive": npos,
                "prevalence": npos / len(data) if data else "",
            })

    # Save.
    write_csv(OUT_DIR / "01_unique_image_multilabel_master.csv", rows, fields)
    write_csv(OUT_DIR / "02_reference_development_5fold.csv", reference, fields)
    write_csv(OUT_DIR / "03_historical_external.csv", historical, fields)
    write_csv(OUT_DIR / "04_field_external.csv", field, fields)
    write_csv(OUT_DIR / "05_reference_quarantine.csv", quarantine, fields)

    balance_fields = ["fold", "n_images", "n_multilineage"] + [f"n_{sid}" for sid in STYLE_IDS]
    write_csv(OUT_DIR / "09_fold_balance.csv", balance_rows, balance_fields)

    write_csv(
        OUT_DIR / "10_domain_lineage_prevalence.csv",
        prevalence_rows,
        ["domain", "style_id", "style_name", "n_domain_images", "n_positive", "prevalence"],
    )

    # Summary.
    md = []
    md.append("# Paper 1 Frozen Dataset Summary — v3")
    md.append("")
    md.append("## Final modeling sets")
    md.append("")
    md.append(f"- Development/reference: **{len(reference)}** unique images")
    md.append(f"- Reference quarantine: **{len(quarantine)}** unique images")
    md.append(f"- Historical external: **{len(historical)}** unique images")
    md.append(f"- Field external: **{len(field)}** unique images")
    md.append("")
    md.append("## Corrected five-fold split")
    md.append("")
    md.append("| Fold | N | Multi | " + " | ".join(STYLE_IDS) + " |")
    md.append("|---:|---:|---:|" + "|".join(["---:"] * 12) + "|")

    for r in balance_rows:
        vals = [r["fold"], r["n_images"], r["n_multilineage"]] + [
            r[f"n_{sid}"] for sid in STYLE_IDS
        ]
        md.append("| " + " | ".join(str(x) for x in vals) + " |")

    md.append("")
    md.append("## QC")
    md.append("")
    md.append("- Cross-domain exact pixel duplicates: **0**.")
    md.append(
        "- Reference records explicitly marked `非目标` or `硬负样本` are quarantined and do not enter supervised model development."
    )
    md.append(
        "- Historical and field domains remain untouched external evaluation sets."
    )

    (OUT_DIR / "00_frozen_dataset_summary_v3.md").write_text(
        "\n".join(md), encoding="utf-8"
    )

    print("=" * 100)
    print("PAPER 1 FREEZE v3 COMPLETE")
    print("=" * 100)
    print(f"Development/reference: {len(reference):,}")
    print(f"Quarantine:            {len(quarantine):,}")
    print(f"Historical external:   {len(historical):,}")
    print(f"Field external:        {len(field):,}")
    print(f"Fold sizes:            {fold_sizes}")
    print()
    for sid in STYLE_IDS:
        vals = [r[f'n_{sid}'] for r in balance_rows]
        print(f"{sid} {STYLE_NAMES[sid]:<14} {vals}")
    print()
    print(f"Output: {OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
