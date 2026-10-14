# -*- coding: utf-8 -*-
"""
LabelMe dataset auditor for the Chinese Eastern Railway architectural-style project.

Purpose
-------
Recursively scans the three user datasets:
1) 《中东铁路大画册》1365p_labelme可打开版
2) 扩充谱系2307p
3) 实拍 1668p

It inventories images and LabelMe JSON files, extracts all internal labels/shapes,
checks source/style balance and annotation problems, and writes compact audit files
that can be uploaded to ChatGPT for the next modeling step.

Dependencies
------------
Required: Python 3.9+ standard library only.
Optional: Pillow (pip install pillow) for image dimensions and perceptual dHash.

Outputs
-------
dataset_audit_output/
    00_dataset_audit.md
    00_dataset_summary.json
    01_source_style_summary.csv
    02_label_summary.csv
    03_label_by_style.csv
    04_label_by_source.csv
    05_image_manifest.csv
    06_json_manifest.csv
    07_shape_records.csv
    08_issues.csv
    09_exact_duplicates.csv
    10_dhash_duplicates.csv
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import csv
import hashlib
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

ROOTS = [
    (
        "historical_album",
        Path(str(PROJECT_ROOT / "annotations" / "historical_album")),
    ),
    (
        "expanded_reference",
        Path(str(PROJECT_ROOT / "annotations" / "expanded_reference")),
    ),
    (
        "field_photographs",
        Path(str(PROJECT_ROOT / "annotations" / "field_photographs")),
    ),
]

OUTPUT_DIR = Path(str(PROJECT_ROOT / "dataset_audit_output"))

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP",
}

# Exact hashes are valuable for preventing train/test leakage.
# For ~5,000 images this is usually acceptable; set False if storage is very slow.
CHECK_EXACT_DUPLICATES = True

# dHash can catch many visually identical images saved/resized differently.
# Requires Pillow. It is not a full near-duplicate detector.
CHECK_DHASH_DUPLICATES = True

# Save one row per LabelMe shape. Usually fine for a few tens of thousands of shapes.
SAVE_SHAPE_RECORDS = True

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

# Matches a real style folder such as:
# "01俄罗斯传统建筑18p", "05折衷主义190p_labelme可打开版", "12_早期现代主义207p"
# but deliberately DOES NOT match a grouping folder like "06-12(1300p)".
STYLE_FOLDER_RE = re.compile(r"^\s*(0[1-9]|1[0-2])(?=[_\s\u4e00-\u9fff])")

# =============================================================================
# 1. OPTIONAL PILLOW
# =============================================================================

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False


# =============================================================================
# 2. HELPERS
# =============================================================================

def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def normalize_label(value: Any) -> str:
    """Conservative normalization: Unicode NFKC + trim + collapse whitespace."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def infer_style(path: Path, root: Path) -> Tuple[str, str, str]:
    """
    Infer style from the nearest matching ancestor folder.
    Returns: style_id, canonical_style_name, raw_style_folder
    """
    current = path.parent
    root_resolved = root.resolve()

    while True:
        name = current.name
        m = STYLE_FOLDER_RE.match(name)
        if m:
            sid = m.group(1)
            return sid, STYLE_NAMES.get(sid, f"style_{sid}"), name

        try:
            if current.resolve() == root_resolved:
                break
        except Exception:
            if current == root:
                break

        if current.parent == current:
            break
        current = current.parent

    return "UNRESOLVED", "UNRESOLVED", ""


def read_json_robust(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str], str]:
    """
    Returns (dict_or_none, error_or_none, encoding_used).
    """
    encodings = ("utf-8-sig", "utf-8", "gb18030")
    last_error = None
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                return None, "Top-level JSON is not an object/dict", enc
            return obj, None, enc
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
    return None, last_error, ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def dhash_file(path: Path, hash_size: int = 8) -> Optional[str]:
    """
    Difference hash. Same hash is a useful first-pass indicator of visually
    identical / near-identical images, but is not proof of identity.
    """
    if not PIL_AVAILABLE:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((hash_size + 1, hash_size))
            pixels = list(im.getdata())
        bits = []
        width = hash_size + 1
        for y in range(hash_size):
            row = y * width
            for x in range(hash_size):
                bits.append(1 if pixels[row + x] > pixels[row + x + 1] else 0)
        value = 0
        for bit in bits:
            value = (value << 1) | bit
        return f"{value:0{hash_size * hash_size // 4}x}"
    except Exception:
        return None


def get_image_size(path: Path) -> Tuple[Optional[int], Optional[int]]:
    if not PIL_AVAILABLE:
        return None, None
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def find_linked_image(json_path: Path, data: Dict[str, Any]) -> Optional[Path]:
    """
    Resolve image referenced by LabelMe JSON:
    1) imagePath relative to JSON folder
    2) same stem + known image extension
    """
    image_path = data.get("imagePath")
    if image_path:
        candidate = Path(str(image_path))
        if not candidate.is_absolute():
            candidate = json_path.parent / candidate
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

        # LabelMe files sometimes store just a Windows basename/path.
        basename = Path(str(image_path).replace("\\", "/")).name
        candidate2 = json_path.parent / basename
        if candidate2.exists() and candidate2.is_file():
            return candidate2.resolve()

    for ext in sorted(IMAGE_EXTS):
        candidate = json_path.with_suffix(ext)
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    return None


def bbox_from_points(points: Any) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not isinstance(points, list) or not points:
        return None, None, None, None
    xs, ys = [], []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            except Exception:
                pass
    if not xs:
        return None, None, None, None
    return min(xs), min(ys), max(xs), max(ys)


def polygon_area(points: Any) -> Optional[float]:
    if not isinstance(points, list) or len(points) < 3:
        return None
    pts = []
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                pts.append((float(p[0]), float(p[1])))
            except Exception:
                return None
    if len(pts) < 3:
        return None
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    def clean(x: Any) -> str:
        s = "" if x is None else str(x)
        return s.replace("|", r"\|").replace("\n", " ")
    out = []
    out.append("| " + " | ".join(clean(x) for x in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(clean(x) for x in row) + " |")
    return "\n".join(out)


# =============================================================================
# 3. MAIN AUDIT
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    issues: List[Dict[str, Any]] = []
    image_manifest: List[Dict[str, Any]] = []
    json_manifest: List[Dict[str, Any]] = []
    shape_records: List[Dict[str, Any]] = []

    # For linking JSON <-> images
    image_lookup: Dict[str, Dict[str, Any]] = {}
    json_to_labels: Dict[str, List[str]] = {}
    json_to_shape_count: Dict[str, int] = {}
    json_to_linked_image: Dict[str, str] = {}

    label_shape_count = Counter()
    label_image_sets = defaultdict(set)
    label_sources = defaultdict(Counter)
    label_styles = defaultdict(Counter)
    label_shape_types = defaultdict(Counter)
    label_raw_variants = defaultdict(Counter)

    source_style_stats = defaultdict(lambda: Counter())
    source_stats = defaultdict(Counter)
    style_stats = defaultdict(Counter)

    exact_hash_groups = defaultdict(list)
    dhash_groups = defaultdict(list)

    root_status = {}

    print("=" * 100)
    print("CHINESE EASTERN RAILWAY LABELME DATASET AUDIT")
    print("=" * 100)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Pillow available: {PIL_AVAILABLE}")
    print()

    # -------------------------------------------------------------------------
    # Pass A: images
    # -------------------------------------------------------------------------
    for source_name, root in ROOTS:
        exists = root.exists()
        root_status[source_name] = {"path": str(root), "exists": exists}
        print(f"[ROOT] {source_name}: {'OK' if exists else 'MISSING'}")
        print(f"       {root}")

        if not exists:
            issues.append({
                "severity": "ERROR",
                "issue_type": "missing_root",
                "source": source_name,
                "style_id": "",
                "path": str(root),
                "details": "Root directory does not exist.",
            })
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in IMAGE_EXTS:
                continue

            sid, sname, raw_folder = infer_style(path, root)
            if sid == "UNRESOLVED":
                issues.append({
                    "severity": "WARNING",
                    "issue_type": "unresolved_style",
                    "source": source_name,
                    "style_id": sid,
                    "path": str(path),
                    "details": "Image is not inside a recognized 01-12 style folder.",
                })

            width, height = get_image_size(path)
            fsize = None
            try:
                fsize = path.stat().st_size
            except Exception:
                pass

            exact_hash = sha256_file(path) if CHECK_EXACT_DUPLICATES else None
            dhash = dhash_file(path) if CHECK_DHASH_DUPLICATES else None

            if exact_hash:
                exact_hash_groups[exact_hash].append(str(path))
            if dhash:
                dhash_groups[dhash].append(str(path))

            row = {
                "source": source_name,
                "style_id": sid,
                "style_name": sname,
                "raw_style_folder": raw_folder,
                "image_path": str(path.resolve()),
                "relative_path": safe_rel(path, root),
                "extension": path.suffix.lower(),
                "file_size_bytes": fsize,
                "width_from_file": width,
                "height_from_file": height,
                "same_stem_json_exists": path.with_suffix(".json").exists(),
                "sha256": exact_hash or "",
                "dhash": dhash or "",
                # populated after JSON pass
                "labelme_json_path": "",
                "annotation_shape_count": 0,
                "annotation_labels": "",
            }
            image_manifest.append(row)
            image_lookup[str(path.resolve()).lower()] = row

            source_style_stats[(source_name, sid, sname)]["images"] += 1
            source_stats[source_name]["images"] += 1
            style_stats[(sid, sname)]["images"] += 1

    print()
    print(f"[PASS A] Found {len(image_manifest):,} image files.")

    # -------------------------------------------------------------------------
    # Pass B: JSON / LabelMe annotations
    # -------------------------------------------------------------------------
    json_files_total = 0
    valid_labelme_total = 0

    for source_name, root in ROOTS:
        if not root.exists():
            continue

        for json_path in root.rglob("*.json"):
            if not json_path.is_file():
                continue
            json_files_total += 1

            sid, sname, raw_folder = infer_style(json_path, root)
            data, error, encoding_used = read_json_robust(json_path)

            base_row = {
                "source": source_name,
                "style_id": sid,
                "style_name": sname,
                "raw_style_folder": raw_folder,
                "json_path": str(json_path.resolve()),
                "relative_path": safe_rel(json_path, root),
                "valid_json": False,
                "is_labelme": False,
                "encoding": encoding_used,
                "labelme_version": "",
                "imagePath": "",
                "linked_image_path": "",
                "imageWidth": "",
                "imageHeight": "",
                "shape_count": 0,
                "unique_label_count": 0,
                "labels": "",
                "shape_types": "",
                "error": error or "",
            }

            source_style_stats[(source_name, sid, sname)]["json_files"] += 1
            source_stats[source_name]["json_files"] += 1
            style_stats[(sid, sname)]["json_files"] += 1

            if data is None:
                json_manifest.append(base_row)
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "malformed_json",
                    "source": source_name,
                    "style_id": sid,
                    "path": str(json_path),
                    "details": error or "Unable to parse JSON.",
                })
                continue

            base_row["valid_json"] = True
            shapes = data.get("shapes")
            is_labelme = isinstance(shapes, list) and ("imagePath" in data or "version" in data)
            base_row["is_labelme"] = is_labelme

            if not is_labelme:
                json_manifest.append(base_row)
                issues.append({
                    "severity": "INFO",
                    "issue_type": "non_labelme_json",
                    "source": source_name,
                    "style_id": sid,
                    "path": str(json_path),
                    "details": "Valid JSON, but it does not look like a LabelMe annotation.",
                })
                continue

            valid_labelme_total += 1
            source_style_stats[(source_name, sid, sname)]["labelme_json"] += 1
            source_stats[source_name]["labelme_json"] += 1
            style_stats[(sid, sname)]["labelme_json"] += 1

            linked_image = find_linked_image(json_path, data)
            linked_image_str = str(linked_image) if linked_image else ""

            width = data.get("imageWidth")
            height = data.get("imageHeight")
            try:
                width_num = float(width) if width not in (None, "") else None
            except Exception:
                width_num = None
            try:
                height_num = float(height) if height not in (None, "") else None
            except Exception:
                height_num = None

            labels_this_json: List[str] = []
            shape_types_this_json = Counter()

            for idx, shape in enumerate(shapes):
                if not isinstance(shape, dict):
                    issues.append({
                        "severity": "WARNING",
                        "issue_type": "invalid_shape_object",
                        "source": source_name,
                        "style_id": sid,
                        "path": str(json_path),
                        "details": f"Shape index {idx} is not a dict/object.",
                    })
                    continue

                raw_label = shape.get("label", "")
                label = normalize_label(raw_label)
                shape_type = normalize_label(shape.get("shape_type") or "polygon")
                points = shape.get("points", [])

                if not label:
                    issues.append({
                        "severity": "WARNING",
                        "issue_type": "empty_label",
                        "source": source_name,
                        "style_id": sid,
                        "path": str(json_path),
                        "details": f"Shape index {idx} has an empty label.",
                    })
                    label = "__EMPTY_LABEL__"

                labels_this_json.append(label)
                shape_types_this_json[shape_type] += 1

                label_shape_count[label] += 1
                label_image_sets[label].add(str(json_path.resolve()))
                label_sources[label][source_name] += 1
                label_styles[label][f"{sid} {sname}"] += 1
                label_shape_types[label][shape_type] += 1
                label_raw_variants[label][str(raw_label)] += 1

                xmin, ymin, xmax, ymax = bbox_from_points(points)
                bbox_area = None
                if None not in (xmin, ymin, xmax, ymax):
                    bbox_area = max(0.0, (xmax - xmin) * (ymax - ymin))

                p_area = polygon_area(points)
                image_area = None
                if width_num and height_num and width_num > 0 and height_num > 0:
                    image_area = width_num * height_num

                bbox_ratio = (bbox_area / image_area) if (bbox_area is not None and image_area) else None
                polygon_ratio = (p_area / image_area) if (p_area is not None and image_area) else None

                if SAVE_SHAPE_RECORDS:
                    shape_records.append({
                        "source": source_name,
                        "style_id": sid,
                        "style_name": sname,
                        "raw_style_folder": raw_folder,
                        "json_path": str(json_path.resolve()),
                        "linked_image_path": linked_image_str,
                        "shape_index": idx,
                        "label_raw": str(raw_label),
                        "label_normalized": label,
                        "shape_type": shape_type,
                        "num_points": len(points) if isinstance(points, list) else "",
                        "group_id": shape.get("group_id"),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "bbox_area_px": bbox_area,
                        "polygon_area_px": p_area,
                        "bbox_area_ratio": bbox_ratio,
                        "polygon_area_ratio": polygon_ratio,
                    })

            unique_labels = sorted(set(labels_this_json))
            base_row.update({
                "labelme_version": data.get("version", ""),
                "imagePath": data.get("imagePath", ""),
                "linked_image_path": linked_image_str,
                "imageWidth": width if width is not None else "",
                "imageHeight": height if height is not None else "",
                "shape_count": len(shapes),
                "unique_label_count": len(unique_labels),
                "labels": " | ".join(unique_labels),
                "shape_types": " | ".join(f"{k}:{v}" for k, v in sorted(shape_types_this_json.items())),
            })

            json_manifest.append(base_row)
            json_key = str(json_path.resolve())
            json_to_labels[json_key] = unique_labels
            json_to_shape_count[json_key] = len(shapes)
            json_to_linked_image[json_key] = linked_image_str

            source_style_stats[(source_name, sid, sname)]["shapes"] += len(shapes)
            source_style_stats[(source_name, sid, sname)]["annotated_json"] += int(len(shapes) > 0)
            source_style_stats[(source_name, sid, sname)]["empty_json"] += int(len(shapes) == 0)
            source_stats[source_name]["shapes"] += len(shapes)
            style_stats[(sid, sname)]["shapes"] += len(shapes)

            if len(shapes) == 0:
                issues.append({
                    "severity": "WARNING",
                    "issue_type": "empty_annotation",
                    "source": source_name,
                    "style_id": sid,
                    "path": str(json_path),
                    "details": "LabelMe JSON contains zero shapes.",
                })

            if linked_image is None:
                issues.append({
                    "severity": "ERROR",
                    "issue_type": "missing_linked_image",
                    "source": source_name,
                    "style_id": sid,
                    "path": str(json_path),
                    "details": f"Could not resolve imagePath={data.get('imagePath')!r} or same-stem image.",
                })
            else:
                lookup_key = str(linked_image).lower()
                if lookup_key in image_lookup:
                    imrow = image_lookup[lookup_key]
                    imrow["labelme_json_path"] = str(json_path.resolve())
                    imrow["annotation_shape_count"] = len(shapes)
                    imrow["annotation_labels"] = " | ".join(unique_labels)
                else:
                    issues.append({
                        "severity": "WARNING",
                        "issue_type": "linked_image_outside_inventory",
                        "source": source_name,
                        "style_id": sid,
                        "path": str(json_path),
                        "details": f"Linked image exists but was not found in scanned image inventory: {linked_image}",
                    })

    print(f"[PASS B] Found {json_files_total:,} JSON files; {valid_labelme_total:,} look like valid LabelMe annotations.")

    # -------------------------------------------------------------------------
    # Pass C: annotation coverage / orphan images
    # -------------------------------------------------------------------------
    for row in image_manifest:
        if not row["labelme_json_path"]:
            issues.append({
                "severity": "INFO",
                "issue_type": "image_without_labelme_json",
                "source": row["source"],
                "style_id": row["style_id"],
                "path": row["image_path"],
                "details": "No linked LabelMe JSON was resolved for this image.",
            })

    # -------------------------------------------------------------------------
    # Pass D: duplicate groups
    # -------------------------------------------------------------------------
    exact_dup_rows = []
    exact_group_no = 0
    if CHECK_EXACT_DUPLICATES:
        for h, paths in exact_hash_groups.items():
            if len(paths) > 1:
                exact_group_no += 1
                for p in paths:
                    exact_dup_rows.append({
                        "duplicate_group": exact_group_no,
                        "sha256": h,
                        "path": p,
                    })
                issues.append({
                    "severity": "WARNING",
                    "issue_type": "exact_duplicate_group",
                    "source": "",
                    "style_id": "",
                    "path": "",
                    "details": f"Exact duplicate group {exact_group_no}: {len(paths)} files.",
                })

    dhash_dup_rows = []
    dhash_group_no = 0
    if CHECK_DHASH_DUPLICATES and PIL_AVAILABLE:
        for h, paths in dhash_groups.items():
            if len(paths) > 1:
                dhash_group_no += 1
                for p in paths:
                    dhash_dup_rows.append({
                        "dhash_group": dhash_group_no,
                        "dhash": h,
                        "path": p,
                    })

    # -------------------------------------------------------------------------
    # Pass E: label-level summary
    # -------------------------------------------------------------------------
    label_summary = []
    for label in sorted(label_shape_count, key=lambda x: (-label_shape_count[x], x)):
        source_counts = label_sources[label]
        style_counts = label_styles[label]
        shape_type_counts = label_shape_types[label]
        raw_variants = label_raw_variants[label]

        label_summary.append({
            "label": label,
            "shape_instances": label_shape_count[label],
            "annotated_images": len(label_image_sets[label]),
            "source_count": len(source_counts),
            "style_count": len(style_counts),
            "sources": " | ".join(f"{k}:{v}" for k, v in sorted(source_counts.items())),
            "styles": " | ".join(f"{k}:{v}" for k, v in sorted(style_counts.items())),
            "shape_types": " | ".join(f"{k}:{v}" for k, v in sorted(shape_type_counts.items())),
            "raw_variants": " | ".join(f"{k!r}:{v}" for k, v in raw_variants.most_common()),
        })

    # Label-by-style matrix
    all_labels = [r["label"] for r in label_summary]
    all_style_keys = [f"{sid} {STYLE_NAMES[sid]}" for sid in sorted(STYLE_NAMES)]
    label_by_style_rows = []
    for label in all_labels:
        row = {"label": label}
        for skey in all_style_keys:
            row[skey] = label_styles[label].get(skey, 0)
        label_by_style_rows.append(row)

    # Label-by-source matrix
    source_names = [x[0] for x in ROOTS]
    label_by_source_rows = []
    for label in all_labels:
        row = {"label": label}
        for src in source_names:
            row[src] = label_sources[label].get(src, 0)
        label_by_source_rows.append(row)

    # Source x style summary
    source_style_rows = []
    for source_name, _root in ROOTS:
        for sid in sorted(STYLE_NAMES):
            sname = STYLE_NAMES[sid]
            c = source_style_stats[(source_name, sid, sname)]
            images = c.get("images", 0)
            labelme = c.get("labelme_json", 0)
            source_style_rows.append({
                "source": source_name,
                "style_id": sid,
                "style_name": sname,
                "images": images,
                "json_files": c.get("json_files", 0),
                "labelme_json": labelme,
                "annotated_json": c.get("annotated_json", 0),
                "empty_json": c.get("empty_json", 0),
                "shapes": c.get("shapes", 0),
                "annotation_coverage_vs_images": round(labelme / images, 4) if images else "",
            })

    # -------------------------------------------------------------------------
    # Pass F: write outputs
    # -------------------------------------------------------------------------

    write_csv(
        OUTPUT_DIR / "01_source_style_summary.csv",
        source_style_rows,
        [
            "source", "style_id", "style_name", "images", "json_files",
            "labelme_json", "annotated_json", "empty_json", "shapes",
            "annotation_coverage_vs_images",
        ],
    )

    write_csv(
        OUTPUT_DIR / "02_label_summary.csv",
        label_summary,
        [
            "label", "shape_instances", "annotated_images", "source_count",
            "style_count", "sources", "styles", "shape_types", "raw_variants",
        ],
    )

    write_csv(
        OUTPUT_DIR / "03_label_by_style.csv",
        label_by_style_rows,
        ["label"] + all_style_keys,
    )

    write_csv(
        OUTPUT_DIR / "04_label_by_source.csv",
        label_by_source_rows,
        ["label"] + source_names,
    )

    write_csv(
        OUTPUT_DIR / "05_image_manifest.csv",
        image_manifest,
        [
            "source", "style_id", "style_name", "raw_style_folder",
            "image_path", "relative_path", "extension", "file_size_bytes",
            "width_from_file", "height_from_file", "same_stem_json_exists",
            "labelme_json_path", "annotation_shape_count", "annotation_labels",
            "sha256", "dhash",
        ],
    )

    write_csv(
        OUTPUT_DIR / "06_json_manifest.csv",
        json_manifest,
        [
            "source", "style_id", "style_name", "raw_style_folder",
            "json_path", "relative_path", "valid_json", "is_labelme", "encoding",
            "labelme_version", "imagePath", "linked_image_path", "imageWidth",
            "imageHeight", "shape_count", "unique_label_count", "labels",
            "shape_types", "error",
        ],
    )

    if SAVE_SHAPE_RECORDS:
        write_csv(
            OUTPUT_DIR / "07_shape_records.csv",
            shape_records,
            [
                "source", "style_id", "style_name", "raw_style_folder",
                "json_path", "linked_image_path", "shape_index", "label_raw",
                "label_normalized", "shape_type", "num_points", "group_id",
                "xmin", "ymin", "xmax", "ymax", "bbox_area_px",
                "polygon_area_px", "bbox_area_ratio", "polygon_area_ratio",
            ],
        )

    write_csv(
        OUTPUT_DIR / "08_issues.csv",
        issues,
        ["severity", "issue_type", "source", "style_id", "path", "details"],
    )

    write_csv(
        OUTPUT_DIR / "09_exact_duplicates.csv",
        exact_dup_rows,
        ["duplicate_group", "sha256", "path"],
    )

    write_csv(
        OUTPUT_DIR / "10_dhash_duplicates.csv",
        dhash_dup_rows,
        ["dhash_group", "dhash", "path"],
    )

    # -------------------------------------------------------------------------
    # Summary JSON
    # -------------------------------------------------------------------------
    total_shapes = sum(label_shape_count.values())
    total_annotated_images = sum(1 for r in image_manifest if r["labelme_json_path"])
    total_unannotated_images = len(image_manifest) - total_annotated_images

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "pillow_available": PIL_AVAILABLE,
        "roots": root_status,
        "totals": {
            "images": len(image_manifest),
            "json_files": json_files_total,
            "valid_labelme_json": valid_labelme_total,
            "images_with_resolved_labelme": total_annotated_images,
            "images_without_resolved_labelme": total_unannotated_images,
            "shape_instances": total_shapes,
            "unique_normalized_labels": len(label_shape_count),
            "issues": len(issues),
            "exact_duplicate_groups": exact_group_no,
            "dhash_duplicate_groups": dhash_group_no,
        },
        "source_totals": {
            src: dict(source_stats[src]) for src, _ in ROOTS
        },
        "style_totals": {
            f"{sid} {sname}": dict(style_stats[(sid, sname)])
            for sid, sname in sorted(STYLE_NAMES.items())
        },
        "top_labels": [
            {
                "label": r["label"],
                "shape_instances": r["shape_instances"],
                "annotated_images": r["annotated_images"],
            }
            for r in label_summary[:100]
        ],
        "issue_type_counts": dict(Counter(i["issue_type"] for i in issues)),
    }

    with (OUTPUT_DIR / "00_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # Human-readable Markdown audit
    # -------------------------------------------------------------------------
    md = []
    md.append("# Chinese Eastern Railway LabelMe Dataset Audit")
    md.append("")
    md.append(f"Generated: {summary['generated_at']}")
    md.append("")
    md.append("## 1. Overall totals")
    md.append("")
    md.append(md_table(
        ["Metric", "Value"],
        [
            ["Images", f"{len(image_manifest):,}"],
            ["JSON files", f"{json_files_total:,}"],
            ["Valid LabelMe JSON", f"{valid_labelme_total:,}"],
            ["Images with resolved LabelMe", f"{total_annotated_images:,}"],
            ["Images without resolved LabelMe", f"{total_unannotated_images:,}"],
            ["Shape instances", f"{total_shapes:,}"],
            ["Unique normalized labels", f"{len(label_shape_count):,}"],
            ["Exact duplicate groups", exact_group_no],
            ["dHash duplicate groups", dhash_group_no if PIL_AVAILABLE else "Pillow unavailable"],
            ["Audit issues", f"{len(issues):,}"],
        ],
    ))
    md.append("")

    md.append("## 2. Root status")
    md.append("")
    md.append(md_table(
        ["Source", "Exists", "Path"],
        [
            [src, root_status[src]["exists"], root_status[src]["path"]]
            for src, _ in ROOTS
        ],
    ))
    md.append("")

    md.append("## 3. Source × style inventory")
    md.append("")
    source_style_md_rows = []
    for r in source_style_rows:
        source_style_md_rows.append([
            r["source"],
            r["style_id"],
            r["style_name"],
            r["images"],
            r["labelme_json"],
            r["shapes"],
            r["annotation_coverage_vs_images"],
        ])
    md.append(md_table(
        ["Source", "ID", "Style", "Images", "LabelMe", "Shapes", "Coverage"],
        source_style_md_rows,
    ))
    md.append("")

    md.append("## 4. Label vocabulary")
    md.append("")
    md.append(f"Unique normalized labels: **{len(label_summary)}**")
    md.append("")
    top_n = min(100, len(label_summary))
    md.append(md_table(
        ["Label", "Shape instances", "Annotated images", "Styles", "Sources", "Shape types"],
        [
            [
                r["label"],
                r["shape_instances"],
                r["annotated_images"],
                r["style_count"],
                r["source_count"],
                r["shape_types"],
            ]
            for r in label_summary[:top_n]
        ],
    ))
    md.append("")

    md.append("## 5. Annotation / file issues")
    md.append("")
    issue_counts = Counter(i["issue_type"] for i in issues)
    if issue_counts:
        md.append(md_table(
            ["Issue type", "Count"],
            [[k, v] for k, v in issue_counts.most_common()],
        ))
    else:
        md.append("No issues detected.")
    md.append("")

    md.append("## 6. Files to upload for research-design freezing")
    md.append("")
    md.append("Please upload at least:")
    md.append("")
    md.append("- `00_dataset_audit.md`")
    md.append("- `01_source_style_summary.csv`")
    md.append("- `02_label_summary.csv`")
    md.append("- `03_label_by_style.csv`")
    md.append("- `08_issues.csv`")
    md.append("")
    md.append("If we need train/test leakage control, also upload:")
    md.append("")
    md.append("- `09_exact_duplicates.csv`")
    md.append("- `10_dhash_duplicates.csv`")
    md.append("")
    md.append("For a full annotation-level audit, upload `07_shape_records.csv` if file size permits.")
    md.append("")

    (OUTPUT_DIR / "00_dataset_audit.md").write_text("\n".join(md), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)
    print(f"Images:                  {len(image_manifest):,}")
    print(f"JSON files:              {json_files_total:,}")
    print(f"Valid LabelMe JSON:      {valid_labelme_total:,}")
    print(f"Resolved annotated imgs: {total_annotated_images:,}")
    print(f"Shape instances:         {total_shapes:,}")
    print(f"Unique labels:           {len(label_shape_count):,}")
    print(f"Issues:                  {len(issues):,}")
    print(f"Exact duplicate groups:  {exact_group_no:,}")
    if PIL_AVAILABLE:
        print(f"dHash duplicate groups:  {dhash_group_no:,}")
    else:
        print("dHash duplicate groups:  skipped (Pillow not installed)")
    print()
    print("Top labels:")
    for r in label_summary[:30]:
        print(
            f"  {r['label']:<30} "
            f"instances={r['shape_instances']:<7} "
            f"images={r['annotated_images']:<7} "
            f"styles={r['style_count']:<3} "
            f"sources={r['source_count']}"
        )
    print()
    print(f"Report folder: {OUTPUT_DIR}")
    print("Upload 00_dataset_audit.md + 01/02/03/08 CSVs to ChatGPT next.")
    print("=" * 100)


if __name__ == "__main__":
    main()
