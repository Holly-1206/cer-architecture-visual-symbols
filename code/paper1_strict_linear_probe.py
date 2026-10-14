# -*- coding: utf-8 -*-
"""Strict external test for the DINOv2 linear probe (complements paper1_strict_external_test.py)."""
from __future__ import annotations
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get("CER_PROJECT_ROOT", Path(__file__).resolve().parents[1]))  # set CER_PROJECT_ROOT or keep the repository layout

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper1_analysis_checks as c  # noqa: E402
from paper1_strict_external_test import outside_flags, boot, N_BOOT, SEED, OUT  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(SEED + 1)
    args = type("Args", (), {})()
    args.data_dir, args.dino_dir, args.out_dir = c.DEFAULT_DATA_DIR, c.DEFAULT_DINO_DIR, OUT
    args.dino_threshold, args.dhash_threshold, args.img_seq_gap = 0.90, 6, 5
    ctx = c.load_context(args)
    df, Y, E = ctx.df, ctx.Y, ctx.emb
    groups = c.build_groups(ctx)
    ref = (df["domain"] == "reference").to_numpy()
    probe = c.fit_linear_probe(E[ref], Y[ref])
    outside, _ = outside_flags(df)
    for domain in ["historical", "field"]:
        dmask = (df["domain"] == domain).to_numpy()
        for subset, mask in [("all", dmask), ("strict", dmask & ~outside)]:
            rows = np.nonzero(mask)[0]
            S = probe.predict_proba(E[rows])
            gid = groups.loc[rows, "group_id"].to_numpy()
            clusters = [np.nonzero(gid == g)[0] for g in np.unique(gid)]
            aucs, aps = c.per_class_scores(Y[rows], S)
            ci, _ = boot(Y[rows], S, clusters, rng, N_BOOT, "cluster")
            print(f"{domain:<10} {subset:<6} n={len(rows):4d}  linear probe macro-AUROC {np.nanmean(aucs):.3f} "
                  f"({ci['macro_AUROC'][0]:.3f}–{ci['macro_AUROC'][1]:.3f})  mAP {np.nanmean(aps):.3f} "
                  f"({ci['mAP'][0]:.3f}–{ci['mAP'][1]:.3f})")


if __name__ == "__main__":
    main()
