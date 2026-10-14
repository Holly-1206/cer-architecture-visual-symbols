# -*- coding: utf-8 -*-
r"""
paper1_strict_external_test.py

Advisor comments C3, C4, C6 (10.12 review)
===========================================

1. Strict external test
   External performance recomputed after removing images that show
   buildings outside the twelve lineages (Chinese temples, palaces, gates,
   Mongolian and Islamic buildings). Such images are identified from the
   annotation record: the file name carries the annotators' note
   "现有分类外" (outside the existing classes) or names a Chinese, Mongolian
   or Islamic building tradition.

2. Taxonomy stress test
   For the out-of-taxonomy images only: which lineage the annotators,
   the classifier and the whole-image prototypes assign them to.

3. The 13 images flagged by building type but not by the explicit note.

Uses frozen data and saved model outputs only; nothing is retrained.

Run:
    python paper1_strict_external_test.py
"""

from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper1_analysis_checks as c  # noqa: E402

OUT = PROJECT_ROOT / "results" / "paper1_strict_external_test_results"
N_BOOT = 2000
SEED = 20261014

EXPLICIT_NOTE = "现有分类外"
TRADITION_PATTERN = r"中国传统|蒙古地区传统|伊斯兰"


def outside_flags(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    explicit = df["stem"].str.contains(EXPLICIT_NOTE).to_numpy()
    tradition = df["stem"].str.contains(TRADITION_PATTERN, regex=True).to_numpy()
    return explicit | tradition, explicit


def boot(Y, S, clusters, rng, n_boot, level):
    vals = {"macro_AUROC": [], "mAP": []}
    for _ in range(n_boot):
        if level == "image":
            idx = rng.integers(0, len(Y), len(Y))
        else:
            pick = rng.integers(0, len(clusters), len(clusters))
            idx = np.concatenate([clusters[p] for p in pick])
        aucs, aps = c.per_class_scores(Y[idx], S[idx])
        vals["macro_AUROC"].append(np.nanmean(aucs))
        vals["mAP"].append(np.nanmean(aps))
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) for k, v in vals.items()}, vals


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    args = type("Args", (), {})()
    args.data_dir, args.dino_dir, args.out_dir = c.DEFAULT_DATA_DIR, c.DEFAULT_DINO_DIR, OUT
    args.dino_threshold, args.dhash_threshold, args.img_seq_gap = 0.90, 6, 5
    ctx = c.load_context(args)
    df, Y, E = ctx.df, ctx.Y, ctx.emb
    groups = c.build_groups(ctx)

    ref = (df["domain"] == "reference").to_numpy()
    P = c.prototypes_from_reference(E[ref], Y[ref])
    outside, explicit = outside_flags(df)

    rows, diff_rows, lineage_rows, report = [], [], [], []

    for domain, pred_name in [
        ("historical", "historical_external_ensemble_predictions.csv"),
        ("field", "field_external_ensemble_predictions.csv"),
    ]:
        dmask = (df["domain"] == domain).to_numpy()
        n_out = int((dmask & outside).sum())
        report.append(f"{domain}: {int(dmask.sum())} images, {n_out} outside the taxonomy "
                      f"({int((dmask & explicit).sum())} with the explicit note)")
        for subset, mask in [("all", dmask), ("strict", dmask & ~outside)]:
            rows_d = np.nonzero(mask)[0]
            Yd = Y[rows_d]
            conv_p, conv_bin = c.load_convnext_external(c.DEFAULT_MODEL_DIR / pred_name, df.loc[rows_d, "pixel_sha256"])
            proto = E[rows_d] @ P.T
            gid = groups.loc[rows_d, "group_id"].to_numpy()
            clusters = [np.nonzero(gid == g)[0] for g in np.unique(gid)]
            n_pos = Yd.sum(0)
            eligible = n_pos >= 15

            for name, S in [("ConvNeXt-Tiny", conv_p), ("DINOv2 prototypes", proto)]:
                aucs, aps = c.per_class_scores(Yd, S)
                ci, _ = boot(Yd, S, clusters, rng, N_BOOT, "cluster")
                rows.append({
                    "domain": domain, "subset": subset, "n_images": len(rows_d), "n_clusters": len(clusters),
                    "method": name,
                    "macro_AUROC": np.nanmean(aucs), "AUROC_low": ci["macro_AUROC"][0], "AUROC_high": ci["macro_AUROC"][1],
                    "mAP": np.nanmean(aps), "mAP_low": ci["mAP"][0], "mAP_high": ci["mAP"][1],
                    "macro_AUROC_lineages_n_ge_15": np.nanmean(aucs[eligible]),
                    "n_lineages_n_ge_15": int(eligible.sum()),
                })
                if name == "ConvNeXt-Tiny":
                    rows[-1]["macro_F1"] = c.threshold_metrics(Yd, conv_bin)["macro_f1"]
                for k, sid in enumerate(c.STYLE_IDS):
                    lineage_rows.append({"domain": domain, "subset": subset, "method": name, "style_id": sid,
                                         "lineage": c.STYLE_NAMES_EN[sid], "n_positive": int(n_pos[k]),
                                         "AUROC": aucs[k], "AP": aps[k]})

            # paired difference, cluster bootstrap
            d_auc, d_ap = [], []
            for _ in range(N_BOOT):
                pick = rng.integers(0, len(clusters), len(clusters))
                idx = np.concatenate([clusters[p] for p in pick])
                a1, p1 = c.per_class_scores(Yd[idx], proto[idx])
                a2, p2 = c.per_class_scores(Yd[idx], conv_p[idx])
                d_auc.append(np.nanmean(a1) - np.nanmean(a2))
                d_ap.append(np.nanmean(p1) - np.nanmean(p2))
            pa, pp = c.per_class_scores(Yd, proto)
            ca, cpp = c.per_class_scores(Yd, conv_p)
            diff_rows.append({
                "domain": domain, "subset": subset,
                "AUROC_difference": np.nanmean(pa) - np.nanmean(ca),
                "AUROC_diff_low": np.quantile(d_auc, 0.025), "AUROC_diff_high": np.quantile(d_auc, 0.975),
                "mAP_difference": np.nanmean(pp) - np.nanmean(cpp),
                "mAP_diff_low": np.quantile(d_ap, 0.025), "mAP_diff_high": np.quantile(d_ap, 0.975),
            })

    perf = pd.DataFrame(rows)
    diffs = pd.DataFrame(diff_rows)
    lineage = pd.DataFrame(lineage_rows)
    perf.to_csv(OUT / "01_external_all_vs_strict.csv", index=False, encoding="utf-8-sig")
    diffs.to_csv(OUT / "02_prototype_minus_convnext_all_vs_strict.csv", index=False, encoding="utf-8-sig")
    lineage.to_csv(OUT / "03_per_lineage_all_vs_strict.csv", index=False, encoding="utf-8-sig")

    # ---- taxonomy stress test ------------------------------------------------
    hist = (df["domain"] == "historical").to_numpy()
    rows_o = np.nonzero(hist & outside)[0]
    conv_p, _ = c.load_convnext_external(c.DEFAULT_MODEL_DIR / "historical_external_ensemble_predictions.csv",
                                          df.loc[rows_o, "pixel_sha256"])
    proto = E[rows_o] @ P.T
    ann = Counter("|".join(s for k, s in enumerate(c.STYLE_IDS) if Y[i, k]) for i in rows_o)
    top_proto = Counter(c.STYLE_IDS[int(np.argmax(proto[j]))] for j in range(len(rows_o)))
    top_conv = Counter(c.STYLE_IDS[int(np.argmax(conv_p[j]))] for j in range(len(rows_o)))
    stress = pd.DataFrame({
        "lineage": [f"{s} {c.STYLE_NAMES_EN[s]}" for s in c.STYLE_IDS],
        "annotators_assigned": [sum(v for k, v in ann.items() if s in k.split("|")) for s in c.STYLE_IDS],
        "prototype_nearest": [top_proto.get(s, 0) for s in c.STYLE_IDS],
        "classifier_top": [top_conv.get(s, 0) for s in c.STYLE_IDS],
    })
    stress.to_csv(OUT / "04_taxonomy_stress_test.csv", index=False, encoding="utf-8-sig")

    # in-taxonomy album images for comparison
    rows_i = np.nonzero(hist & ~outside)[0]
    proto_i = E[rows_i] @ P.T
    in_tax_08 = Counter(c.STYLE_IDS[int(np.argmax(proto_i[j]))] for j in range(len(rows_i))).get("08", 0)

    # ---- the 13 images -------------------------------------------------------
    thirteen = df[(df["domain"] == "historical").to_numpy() & outside & ~explicit][["stem", "style_ids"]]
    thirteen.to_csv(OUT / "05_flagged_by_building_type_only.csv", index=False, encoding="utf-8-sig")

    # ---- report ----------------------------------------------------------------
    lines = ["# Strict external test and taxonomy stress test", ""]
    lines += [f"- {r}" for r in report]
    lines += ["", "## External performance, all vs strict (cluster bootstrap, 2,000 replicates)", "",
              perf.round(3).to_markdown(index=False), "",
              "## Prototypes minus ConvNeXt", "", diffs.round(3).to_markdown(index=False), "",
              f"## Taxonomy stress test: {len(rows_o)} album images outside the twelve lineages", "",
              stress.to_markdown(index=False), "",
              f"Share nearest the Chinese Baroque prototype: {top_proto.get('08', 0)}/{len(rows_o)} "
              f"({100 * top_proto.get('08', 0) / len(rows_o):.0f}%); classifier top = Chinese Baroque: "
              f"{top_conv.get('08', 0)}/{len(rows_o)} ({100 * top_conv.get('08', 0) / len(rows_o):.0f}%). "
              f"In-taxonomy album images nearest Chinese Baroque: {in_tax_08}/{len(rows_i)} "
              f"({100 * in_tax_08 / len(rows_i):.0f}%).", "",
              "## Images flagged by building type but without the explicit note", "",
              thirteen.to_markdown(index=False)]
    (OUT / "00_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
