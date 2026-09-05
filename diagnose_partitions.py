"""
diagnose_partitions.py — inspect per-client label / modality composition
for specific (n_clients, JSD-or-HD target, seed) combinations.

Run this on the same machine/cache as train_cremad.py:

    python diagnose_partitions.py --cremad-path ./CREMA-D --n-clients 6 --kind jsd --target 0.48
    python diagnose_partitions.py --cremad-path ./CREMA-D --n-clients 6 --kind hd  --target 0.70

It reuses build_config / load_cremad / find_alpha_for_jsd / find_alpha_for_hd /
partition_data_fedartml directly from train_cremad.py, so it reconstructs the
EXACT same partitions your training run used (same seeds, same alpha search).

What it prints, per seed:
  - the alpha_modal the search landed on
  - per client: sample count, per-class label histogram, and (if the
    ModalityHeterogeneity "masks" field is in the format we expect) the
    fraction of samples where each modality is actually present.
  - a "<-- CHECK" flag on any client that's small, missing a class entirely,
    or has <10% availability of a modality.

IMPORTANT: the exact shape/meaning of the "masks" array depends on your
fl_modality_heterogeneity.py implementation, which wasn't in what you shared
with me. The script prints masks.shape and a couple of raw rows the FIRST
time it runs so you can eyeball whether the img_avail/aud_avail numbers make
sense before trusting them. If the shape doesn't match what's assumed below
(masks[:, 0] = image, masks[:, 1] = audio), adjust the two lines marked
"ADJUST IF NEEDED".
"""

import argparse
import numpy as np
import torch

from train_cremad import (
    build_config, load_cremad, find_alpha_for_jsd, find_alpha_for_hd,
    partition_data_fedartml, _fedartml_safe_seed,
)
from fl_modality_heterogeneity import ModalityHeterogeneity


def diagnose(n_clients, kind, target, cfg, img_tr, aud_tr, lbl_tr, print_raw_mask_once=True):
    print(f"\n{'='*70}\n n_clients={n_clients}   {kind.upper()}_target={target}\n{'='*70}")
    printed_raw = not print_raw_mask_once

    for seed in cfg.SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        finder = find_alpha_for_jsd if kind == "jsd" else find_alpha_for_hd
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

        print(f"\n  seed={seed}  alpha_modal={alpha_modal:.4f}  label_JSD={label_jsd:.4f}")

        for i in range(n_clients):
            cname = f"client_{i+1}"
            s, e = split_starts[i], split_ends[i]
            y = joint_data[cname]["y"][s:e]
            masks = joint_data[cname]["masks"][s:e]

            if not printed_raw:
                print(f"    [debug] masks.shape={masks.shape}  masks[:3]={masks[:3]}")
                printed_raw = True

            counts = np.bincount(y, minlength=cfg.NUM_CLASSES)
            n_total = len(y)

            img_avail = aud_avail = None
            try:
                # ADJUST IF NEEDED: assumes masks is (n, 2) boolean/0-1,
                # column order matching modality_names=["image","audio"]
                img_avail = float(np.mean(masks[:, 0]))
                aud_avail = float(np.mean(masks[:, 1]))
            except Exception:
                pass

            bad = (
                n_total < 10
                or counts.min() == 0
                or (img_avail is not None and img_avail < 0.10)
                or (aud_avail is not None and aud_avail < 0.10)
            )
            flag = "  <-- CHECK" if bad else ""
            avail_str = (f"  img_avail={img_avail:.2f}  aud_avail={aud_avail:.2f}"
                         if img_avail is not None else "")
            print(f"    {cname}: n={n_total:4d}  class_counts={counts.tolist()}{avail_str}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad-path", default="./CREMA-D")
    ap.add_argument("--cache-path", default="./cremad_features")
    ap.add_argument("--images-dir", default="./images_cremad_v6")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-clients", type=int, required=True)
    ap.add_argument("--kind", choices=["jsd", "hd"], required=True)
    ap.add_argument("--target", type=float, required=True)
    args = ap.parse_args()

    cfg = build_config(args)
    img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te = load_cremad(
        cfg.CREMAD_PATH, cfg.CACHE_PATH, cfg)

    diagnose(args.n_clients, args.kind, args.target, cfg, img_tr, aud_tr, lbl_tr)


if __name__ == "__main__":
    main()