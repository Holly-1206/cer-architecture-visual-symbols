# -*- coding: utf-8 -*-
r"""
paper1_annotation_agreement.py

Paper 1 — inter-annotator agreement for lineage and symbol annotation
(advisor comment C7)
=====================================================================

No record of the fourth reviewer's independent judgements is stored with the
annotation files, so agreement is measured on a fresh blind sample.

STEP 1  sample
    Draws a stratified random sample (default 200 images: proportional to
    domain, at least --min-per-lineage positives per lineage where available)
    and writes a BLIND coding sheet. The sheet shows only the image; the
    original lineage and symbol labels are hidden in a separate key file.

        python paper1_annotation_agreement.py sample

    Output (paper1_annotation_agreement\):
        01_blind_coding_sheet.xlsx   reviewer fills this
        01_blind_coding_sheet.csv    same content, CSV
        02_sample_key_DO_NOT_SHARE.csv
        03_sheet_images\             numbered copies of the sampled images
        04_contact_sheet_XX.jpg      numbered thumbnails for quick viewing

    Reviewer instructions (also written to 00_README.txt):
      - For each image, put 1 in every lineage column y_01 ... y_12 that
        applies (multi-label allowed), leave the others empty or 0.
      - In `AD_symbols`, list the A–D symbol codes present in the image,
        separated by '|' (e.g. A4|B2|C12). Presence only, no counts.
      - Do not open the original LabelMe files while coding.

STEP 2  score
    After the reviewer returns the sheet as 01_blind_coding_sheet_FILLED.xlsx
    (or .csv) in the same folder:

        python paper1_annotation_agreement.py score

    Output:
        10_lineage_kappa.csv          Cohen's kappa per lineage, with bootstrap CI
        11_symbol_kappa.csv           Cohen's kappa per A–D symbol (presence)
        12_agreement_summary.csv      pooled statistics
        13_disagreements.csv          images where lineage sets differ
        00_agreement_report.md        text ready for Section 3.3

Statistics
----------
- Cohen's kappa per lineage on the binary present/absent decision.
- Mean kappa across lineages (macro) and pooled kappa over all
  image-lineage decisions (micro).
- Exact-match rate of lineage sets and mean Jaccard of lineage sets.
- Krippendorff's alpha (nominal) on the pooled binary decisions.
- Same per-symbol and pooled statistics for A–D symbol presence.
- 95% bootstrap CIs (resampling images).

Dependencies
------------
pip install numpy pandas scikit-learn pillow openpyxl
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper1_analysis_checks import (  # noqa: E402
    AD_SYMBOLS,
    DEFAULT_DATA_DIR,
    ROLE_TO_DOMAIN,
    STYLE_IDS,
    STYLE_NAMES_EN,
    open_rgb,
    parse_counts,
    save_contact_sheet,
)

DEFAULT_OUT_DIR = Path(
    str(PROJECT_ROOT / "results" / "paper1_annotation_agreement")
)
SEED = 20261011
LINEAGE_COLS = [f"y_{sid}" for sid in STYLE_IDS]


# =============================================================================
# 1. SAMPLE
# =============================================================================

def load_master(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "01_unique_image_multilabel_master.csv", encoding="utf-8-sig")
    df = df[df["dataset_role"].isin(ROLE_TO_DOMAIN)].copy().reset_index(drop=True)
    df["domain"] = df["dataset_role"].map(ROLE_TO_DOMAIN)
    return df


def draw_sample(df: pd.DataFrame, n: int, min_per_lineage: int, seed: int) -> pd.DataFrame:
    """
    Proportional-by-domain random sample, topped up so that every lineage has
    at least `min_per_lineage` positives (or all available), so that per-lineage
    kappa is estimable for rare lineages.
    """
    rng = np.random.default_rng(seed)
    chosen: set[int] = set()

    # Guarantee minimum positives per lineage (rarest lineages first).
    order = sorted(STYLE_IDS, key=lambda s: df[f"y_{s}"].sum())
    for sid in order:
        have = sum(int(df.at[i, f"y_{sid}"]) for i in chosen)
        need = max(0, min_per_lineage - have)
        pool = [i for i in df.index[df[f"y_{sid}"] == 1] if i not in chosen]
        if need and pool:
            chosen.update(rng.choice(pool, size=min(need, len(pool)), replace=False).tolist())

    # Fill the remainder proportionally to domain size.
    remaining = max(0, n - len(chosen))
    shares = df["domain"].value_counts(normalize=True)
    for domain, share in shares.items():
        pool = [i for i in df.index[df["domain"] == domain] if i not in chosen]
        k = min(len(pool), int(round(remaining * share)))
        if k:
            chosen.update(rng.choice(pool, size=k, replace=False).tolist())

    sample = df.loc[sorted(chosen)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample.insert(0, "item_no", np.arange(1, len(sample) + 1))
    return sample


def run_sample(args) -> None:
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / "03_sheet_images"
    img_dir.mkdir(exist_ok=True)

    df = load_master(args.data_dir)
    sample = draw_sample(df, args.n, args.min_per_lineage, args.seed)

    # Key with original labels (hidden from reviewer).
    key = sample[["item_no", "domain", "pixel_sha256", "image_path", "style_ids", "symbol_counts_json"] + LINEAGE_COLS].copy()
    key["AD_symbols_original"] = key["symbol_counts_json"].map(
        lambda v: "|".join(s for s in AD_SYMBOLS if parse_counts(v).get(s, 0) > 0)
    )
    key.drop(columns=["symbol_counts_json"]).to_csv(out / "02_sample_key_DO_NOT_SHARE.csv", index=False, encoding="utf-8-sig")

    # Blind sheet: numbered image copy, empty label columns.
    rows = []
    for _, r in sample.iterrows():
        src = Path(r["image_path"])
        dst = img_dir / f"{int(r['item_no']):03d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        row = {"item_no": int(r["item_no"]), "image_file": dst.name}
        row.update({c: "" for c in LINEAGE_COLS})
        row["AD_symbols"] = ""
        row["reviewer_notes"] = ""
        rows.append(row)
    sheet = pd.DataFrame(rows)
    header = {f"y_{s}": f"y_{s} {STYLE_NAMES_EN[s]}" for s in STYLE_IDS}
    sheet.rename(columns=header).to_csv(out / "01_blind_coding_sheet.csv", index=False, encoding="utf-8-sig")
    try:
        sheet.rename(columns=header).to_excel(out / "01_blind_coding_sheet.xlsx", index=False)
    except Exception as exc:  # openpyxl missing
        print(f"[WARN] xlsx not written ({exc}); use the CSV.")

    per_sheet = 30
    for start in range(0, len(sample), per_sheet):
        chunk = sample.iloc[start:start + per_sheet]
        save_contact_sheet(
            [open_rgb(p) for p in chunk["image_path"]],
            [f"#{int(i)}" for i in chunk["item_no"]],
            out / f"04_contact_sheet_{start // per_sheet + 1:02d}.jpg",
            5, (320, 260),
        )

    (out / "00_README.txt").write_text(
        "Blind re-annotation for inter-annotator agreement (Paper 1, comment C7)\n\n"
        "Give the reviewer ONLY: 01_blind_coding_sheet.xlsx, 03_sheet_images\\, 04_contact_sheet_*.jpg,\n"
        "the coding notebook and the reference atlas. Do NOT give 02_sample_key_DO_NOT_SHARE.csv.\n\n"
        "For each item:\n"
        "  - put 1 in every lineage column that applies (several allowed);\n"
        "  - list A–D symbols present in AD_symbols, separated by '|', e.g. A4|B2|C12.\n\n"
        "Save the completed sheet as 01_blind_coding_sheet_FILLED.xlsx (or .csv) in this folder, then run:\n"
        "  python paper1_annotation_agreement.py score\n",
        encoding="utf-8",
    )

    counts = sample[LINEAGE_COLS].sum()
    print(f"Sample: {len(sample)} images")
    print(sample["domain"].value_counts().to_string())
    print("Positives per lineage:", " ".join(f"{c[2:]}:{int(v)}" for c, v in counts.items()))
    print(f"Output: {out}")


# =============================================================================
# 2. SCORE
# =============================================================================

def read_filled(out: Path) -> pd.DataFrame:
    for name in ("01_blind_coding_sheet_FILLED.xlsx", "01_blind_coding_sheet_FILLED.csv"):
        p = out / name
        if p.exists():
            sheet = pd.read_excel(p) if p.suffix == ".xlsx" else pd.read_csv(p, encoding="utf-8-sig")
            sheet.columns = [str(c).split(" ")[0] for c in sheet.columns]
            return sheet
    raise FileNotFoundError(f"No filled sheet found in {out}")


def to_binary(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(float).gt(0).astype(int).to_numpy()


def parse_symbol_list(value) -> set[str]:
    if not isinstance(value, str):
        return set()
    parts = [p.strip().upper() for p in value.replace(",", "|").replace(";", "|").replace(" ", "|").split("|")]
    return {p for p in parts if p in AD_SYMBOLS}


def safe_kappa(a: np.ndarray, b: np.ndarray) -> float:
    if len(np.unique(np.r_[a, b])) < 2:
        return np.nan
    return float(cohen_kappa_score(a, b))


def krippendorff_alpha_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Nominal Krippendorff's alpha for two coders, no missing values."""
    values = np.r_[a, b]
    n = len(values)
    p1 = values.mean()
    expected = 2 * p1 * (1 - p1) * n / (n - 1)
    observed = np.mean(a != b)
    return float(1 - observed / expected) if expected > 0 else np.nan


def bootstrap_ci(stat, n_items: int, n_boot: int, rng) -> tuple[float, float]:
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, n_items)
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run_score(args) -> None:
    out = args.out_dir
    rng = np.random.default_rng(args.seed)
    key = pd.read_csv(out / "02_sample_key_DO_NOT_SHARE.csv", encoding="utf-8-sig", keep_default_na=False)
    sheet = read_filled(out)
    m = key.merge(sheet, on="item_no", suffixes=("_orig", "_rev"), validate="one_to_one")
    n = len(m)

    A = np.column_stack([to_binary(m[f"{c}_orig"]) for c in LINEAGE_COLS])
    B = np.column_stack([to_binary(m[f"{c}_rev"]) for c in LINEAGE_COLS])

    unlabeled = int((B.sum(axis=1) == 0).sum())
    if unlabeled:
        print(f"[WARN] {unlabeled} items have no lineage from the reviewer; they count as disagreement.")

    # ---- lineage ------------------------------------------------------------
    lin_rows = []
    for k, sid in enumerate(STYLE_IDS):
        a, b = A[:, k], B[:, k]
        lo, hi = bootstrap_ci(lambda idx: safe_kappa(a[idx], b[idx]), n, args.n_boot, rng)
        lin_rows.append({
            "style_id": sid,
            "style_name_en": STYLE_NAMES_EN[sid],
            "n_original_positive": int(a.sum()),
            "n_reviewer_positive": int(b.sum()),
            "percent_agreement": float(np.mean(a == b)),
            "cohen_kappa": safe_kappa(a, b),
            "ci95_low": lo,
            "ci95_high": hi,
        })
    lin = pd.DataFrame(lin_rows)
    lin.to_csv(out / "10_lineage_kappa.csv", index=False, encoding="utf-8-sig")

    def macro_kappa(idx):
        return float(np.nanmean([safe_kappa(A[idx, k], B[idx, k]) for k in range(12)]))

    def pooled_kappa(idx):
        return safe_kappa(A[idx].ravel(), B[idx].ravel())

    def jaccard_sets(idx):
        inter = np.logical_and(A[idx], B[idx]).sum(axis=1)
        union = np.logical_or(A[idx], B[idx]).sum(axis=1)
        return float(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 1.0)))

    summary = []
    for name, fn in [
        ("lineage_macro_kappa", macro_kappa),
        ("lineage_pooled_kappa", pooled_kappa),
        ("lineage_set_exact_match", lambda idx: float(np.mean((A[idx] == B[idx]).all(axis=1)))),
        ("lineage_set_mean_jaccard", jaccard_sets),
        ("lineage_krippendorff_alpha", lambda idx: krippendorff_alpha_binary(A[idx].ravel(), B[idx].ravel())),
    ]:
        lo, hi = bootstrap_ci(fn, n, args.n_boot, rng)
        summary.append({"statistic": name, "value": fn(np.arange(n)), "ci95_low": lo, "ci95_high": hi})

    # ---- symbols ------------------------------------------------------------
    has_symbols = "AD_symbols" in m and m["AD_symbols"].astype(str).str.strip().replace("nan", "").ne("").any()
    sym = pd.DataFrame()
    if has_symbols:
        SA = np.array([[s in parse_symbol_list(v) for s in AD_SYMBOLS] for v in m["AD_symbols_original"]], dtype=int)
        SB = np.array([[s in parse_symbol_list(v) for s in AD_SYMBOLS] for v in m["AD_symbols"]], dtype=int)
        sym_rows = []
        for j, code in enumerate(AD_SYMBOLS):
            a, b = SA[:, j], SB[:, j]
            if a.sum() + b.sum() == 0:
                continue
            sym_rows.append({
                "symbol_code": code,
                "n_original_present": int(a.sum()),
                "n_reviewer_present": int(b.sum()),
                "percent_agreement": float(np.mean(a == b)),
                "cohen_kappa": safe_kappa(a, b),
            })
        sym = pd.DataFrame(sym_rows)
        sym.to_csv(out / "11_symbol_kappa.csv", index=False, encoding="utf-8-sig")

        frequent = sym["n_original_present"] + sym["n_reviewer_present"] >= 10
        for name, fn in [
            ("symbol_pooled_kappa", lambda idx: safe_kappa(SA[idx].ravel(), SB[idx].ravel())),
            ("symbol_krippendorff_alpha", lambda idx: krippendorff_alpha_binary(SA[idx].ravel(), SB[idx].ravel())),
        ]:
            lo, hi = bootstrap_ci(fn, n, args.n_boot, rng)
            summary.append({"statistic": name, "value": fn(np.arange(n)), "ci95_low": lo, "ci95_high": hi})
        summary.append({
            "statistic": "symbol_median_kappa_frequent(>=10 marks)",
            "value": float(sym.loc[frequent, "cohen_kappa"].median()) if frequent.any() else np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
        })

    summ = pd.DataFrame(summary)
    summ.to_csv(out / "12_agreement_summary.csv", index=False, encoding="utf-8-sig")

    diff = m[(A != B).any(axis=1)][["item_no", "domain", "style_ids", "image_path"]].copy()
    diff["reviewer_style_ids"] = ["|".join(s for k, s in enumerate(STYLE_IDS) if B[i, k]) for i in np.nonzero((A != B).any(axis=1))[0]]
    diff.to_csv(out / "13_disagreements.csv", index=False, encoding="utf-8-sig")

    # ---- report -------------------------------------------------------------
    s = summ.set_index("statistic")
    def fmt(name):
        r = s.loc[name]
        return f"{r['value']:.2f} (95% CI {r['ci95_low']:.2f}–{r['ci95_high']:.2f})"

    lines = [
        "# Inter-annotator agreement (C7)",
        "",
        f"Blind re-annotation of {n} images "
        f"({', '.join(f'{d} {c}' for d, c in m['domain'].value_counts().items())}).",
        "",
        summ.round(3).to_markdown(index=False),
        "",
        lin.round(3).to_markdown(index=False),
        "",
        "## Draft sentence for Section 3.3",
        "",
        f"To measure agreement, the reviewer re-annotated a stratified random sample of {n} images "
        f"without seeing the original labels. Agreement on lineage attribution was "
        f"κ = {fmt('lineage_pooled_kappa')} over all image–lineage decisions "
        f"(mean per-lineage κ = {s.loc['lineage_macro_kappa', 'value']:.2f}; "
        f"per-lineage range {lin['cohen_kappa'].min():.2f}–{lin['cohen_kappa'].max():.2f})"
        + (f", and on the presence of A–D symbols κ = {fmt('symbol_pooled_kappa')}." if has_symbols else "."),
    ]
    (out / "00_agreement_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description="Inter-annotator agreement for Paper 1 (C7).")
    p.add_argument("step", choices=["sample", "score"])
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--min-per-lineage", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    if args.step == "sample":
        run_sample(args)
    else:
        run_score(args)


if __name__ == "__main__":
    main()
