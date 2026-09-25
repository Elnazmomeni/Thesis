"""
diagnose_sweep_summary.py — run the degenerate-client check across the FULL
client sweep (all client counts x all JSD levels x all HD levels), instead of
one combo at a time, and report just the summary numbers you actually need
for the thesis: what fraction of clients are individually broken at each
operating point.

"Broken" (flagged) means, per client:
    - fewer than --min-samples total samples, OR
    - at least one emotion class with zero examples, OR
    - image availability < --avail-threshold, OR
    - audio availability < --avail-threshold

Usage
-----
    python diagnose_sweep_summary.py --cremad-path ./CREMA-D

    # restrict to a subset (faster / for a quick check):
    python diagnose_sweep_summary.py --cremad-path ./CREMA-D \\
        --client-counts 4 6 10 --jsd-levels 0.02 0.24 0.48 --hd-levels 0.05 0.37 0.70

Output
------
Prints a summary table (one row per n_clients x kind x level) with the
flagged-client fraction, averaged over the 5 seeds, plus per-seed detail.
Also writes a CSV (--out) with one row per (n_clients, kind, level, seed) so
you can pivot/plot it however you want for the thesis appendix.

Cost note: for each (kind, level, n_clients) combo not already in the
in-process alpha cache, this triggers one alpha search (~70 candidates x 5
probe seeds of build_client_datasets — partitioning + heterogeneity scoring
only, NO FedAvg training, so it's fast) plus 5 final builds. Running the
full default grid (5 client counts x 5 JSD levels x 5 HD levels) is 50
combos total; expect it to take a while but nowhere near as long as
retraining, since no neural net is ever touched here.
"""

import argparse
import csv
import numpy as np
import torch

from train_cremad import (
    build_config, load_cremad, find_alpha_for_jsd, find_alpha_for_hd,
    partition_data_fedartml, _fedartml_safe_seed,
)
from fl_modality_heterogeneity import ModalityHeterogeneity


def check_partition(n_clients, kind, target, cfg, img_tr, aud_tr, lbl_tr,
                     min_samples, avail_threshold):
    """Returns list of dicts, one per seed, with flagged-client stats."""
    rows = []
    finder = find_alpha_for_jsd if kind == "jsd" else find_alpha_for_hd

    for seed in cfg.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        alpha_modal = finder(target, n_clients, seed, cfg, img_tr, aud_tr, lbl_tr)

        rs = _fedartml_safe_seed(seed, cfg)
        partitions, label_jsd, label_hd = partition_data_fedartml(
            img_tr, aud_tr, lbl_tr, num_clients=n_clients,
            alpha_label=cfg.ALPHA_LABEL_FIXED, cfg=cfg, random_state=rs)

        mh = ModalityHeterogeneity(modality_names=["image", "audio"], random_state=rs)
        all_img = np.concatenate([p[0] for p in partitions], axis=0)
        all_aud = np.concatenate([p[1] for p in partitions], axis=0)
        all_lbl = np.concatenate([p[2] for p in partitions], axis=0)

        split_sizes = [len(p[2]) for p in partitions]
        split_ends = np.cumsum(split_sizes)
        split_starts = np.concatenate([[0], split_ends[:-1]])

        joint_data, _ = mh.assign_modalities_to_clients(
            modality_arrays={"image": all_img, "audio": all_aud},
            y=all_lbl, num_clients=n_clients, alpha=alpha_modal, prefix_cli="client")

        n_flagged = 0
        flagged_reasons = []
        for i in range(n_clients):
            cname = f"client_{i+1}"
            s, e = split_starts[i], split_ends[i]
            y = joint_data[cname]["y"][s:e]
            masks = joint_data[cname]["masks"][s:e]

            counts = np.bincount(y, minlength=cfg.NUM_CLASSES)
            n_total = len(y)
            img_avail = float(np.mean(masks[:, 0])) if n_total else 0.0
            aud_avail = float(np.mean(masks[:, 1])) if n_total else 0.0

            reasons = []
            if n_total < min_samples:
                reasons.append("tiny_n")
            if counts.min() == 0:
                reasons.append("missing_class")
            if img_avail < avail_threshold:
                reasons.append("no_image")
            if aud_avail < avail_threshold:
                reasons.append("no_audio")

            if reasons:
                n_flagged += 1
                flagged_reasons.extend(reasons)

        rows.append({
            "n_clients": n_clients,
            "kind": kind,
            "target": target,
            "seed": seed,
            "alpha_modal": alpha_modal,
            "label_jsd": label_jsd,
            "label_hd": label_hd,
            "n_flagged": n_flagged,
            "n_total_clients": n_clients,
            "flagged_frac": n_flagged / n_clients,
            "reasons": ";".join(sorted(set(flagged_reasons))),
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad-path", default="./CREMA-D")
    ap.add_argument("--cache-path", default="./cremad_features")
    ap.add_argument("--images-dir", default="./images_cremad_v6")
    ap.add_argument("--device", default=None)
    ap.add_argument("--client-counts", type=int, nargs="+", default=None,
                     help="Defaults to the full CLIENT_SWEEP from train_cremad.py: 4 6 10 20 100")
    ap.add_argument("--jsd-levels", type=float, nargs="+", default=None,
                     help="Defaults to FIXED_JSD_LEVELS: 0.02 0.10 0.24 0.39 0.48")
    ap.add_argument("--hd-levels", type=float, nargs="+", default=None,
                     help="Defaults to FIXED_HD_LEVELS: 0.05 0.16 0.37 0.57 0.70")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--avail-threshold", type=float, default=0.10)
    ap.add_argument("--out", default="./partition_diagnostic_summary.csv")
    args = ap.parse_args()

    cfg = build_config(args)
    client_counts = args.client_counts or cfg.CLIENT_SWEEP
    jsd_levels = args.jsd_levels if args.jsd_levels is not None else [0.02, 0.10, 0.24, 0.39, 0.48]
    hd_levels = args.hd_levels if args.hd_levels is not None else [0.05, 0.16, 0.37, 0.57, 0.70]

    img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te = load_cremad(
        cfg.CREMAD_PATH, cfg.CACHE_PATH, cfg)

    all_rows = []
    combos = (
        [(n, "jsd", j) for n in client_counts for j in jsd_levels]
        + [(n, "hd", h) for n in client_counts for h in hd_levels]
    )
    total = len(combos)

    print(f"\nRunning {total} (n_clients, kind, level) combos "
          f"x {len(cfg.SEEDS)} seeds each ...\n")

    for i, (n_clients, kind, level) in enumerate(combos, 1):
        print(f"[{i}/{total}] n_clients={n_clients:<4} {kind.upper()}={level:.2f} ...", end=" ", flush=True)
        rows = check_partition(n_clients, kind, level, cfg, img_tr, aud_tr, lbl_tr,
                                args.min_samples, args.avail_threshold)
        all_rows.extend(rows)
        mean_frac = float(np.mean([r["flagged_frac"] for r in rows]))
        print(f"mean flagged = {mean_frac:.0%}")

        # incremental save so a Ctrl-C or crash doesn't lose everything
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"\nSaved per-seed detail to: {args.out}\n")

    # ── summary table (mean flagged % per n_clients x kind x level) ────────
    print(f"{'='*78}\nSUMMARY — mean %% of clients flagged (over {len(cfg.SEEDS)} seeds)\n{'='*78}")
    print(f"{'n_clients':<10}{'kind':<6}{'target':<8}{'mean_flagged%':<15}{'seed_range'}")
    seen = set()
    for n_clients, kind, level in combos:
        key = (n_clients, kind, round(level, 4))
        if key in seen:
            continue
        seen.add(key)
        matching = [r for r in all_rows if r["n_clients"] == n_clients and r["kind"] == kind
                    and abs(r["target"] - level) < 1e-6]
        fracs = [r["flagged_frac"] for r in matching]
        print(f"{n_clients:<10}{kind:<6}{level:<8.2f}{np.mean(fracs):<15.0%}"
              f"[{min(fracs):.0%} - {max(fracs):.0%}]")

    print(f"\nFull per-seed CSV: {args.out}")
    print("Columns: n_clients, kind, target, seed, alpha_modal, label_jsd, label_hd, "
          "n_flagged, n_total_clients, flagged_frac, reasons")


if __name__ == "__main__":
    main()