"""
Re-runs ONLY alpha_modal = 3 (all 5 seeds) using the patched train_fedavg
(round-wise LR decay), so you can compare it against the existing
no-decay baseline for alpha=3 already sitting in your checkpoint pkl.

Usage (same CLI args as your original script):
    python rerun_alpha3_with_decay.py --ravdess-path ./RAVDESS \
        --cache-path ./ravdess_features

Requires train_ravdess.py (patched version, with round-wise LR decay
added inside train_fedavg) to be importable from the same directory.
"""
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import everything from the PATCHED script (must be named train_ravdess.py
# in the same folder, with the round_lr()/cosine-decay edit applied).
from train_ravdess import (
    build_config, load_ravdess, make_tensor_dataset, build_client_datasets,
    train_fedavg, _fedartml_safe_seed,
)

BASELINE_ALPHA3 = {  # from your existing no-decay checkpoint, for reference
    "f1":  [0.5795788662996457, 0.4342889209614992, 0.8114496292856227,
            0.8329391687190579, 0.7654098339804578],
    "acc": [0.5771929824561404, 0.4473684210526316, 0.8122807017543859,
            0.8350877192982457, 0.7666666666666667],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess-path", default="./RAVDESS")
    parser.add_argument("--cache-path", default="./ravdess_features")
    parser.add_argument("--images-dir", default="./images_ravdess_100")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="./alpha3_decay_result.pkl")
    args = parser.parse_args()

    cfg = build_config(args)
    torch.manual_seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)
    (img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te) = load_ravdess(
        cfg.RAVDESS_PATH, cfg.CACHE_PATH, cfg)

    test_loader = DataLoader(make_tensor_dataset(img_te, aud_te, lbl_te),
                              batch_size=8, shuffle=False, num_workers=0)

    alpha_modal = 3
    seed_f1s, seed_accs = [], []

    for seed in tqdm(cfg.SEEDS, desc=f"alpha={alpha_modal} (with LR decay)"):
        torch.manual_seed(seed)
        np.random.seed(seed)

        (client_ds, modal_jsd, modal_hd, label_jsd, label_hd, _) = build_client_datasets(
            img_tr, aud_tr, lbl_tr,
            alpha_modal=alpha_modal,
            cfg=cfg,
            alpha_label=cfg.ALPHA_LABEL_FIXED,
            num_clients=cfg.NUM_CLIENTS,
            random_state=_fedartml_safe_seed(seed, cfg))

        fl_f1, fl_acc = train_fedavg(client_ds, test_loader, cfg,
                                      fl_rounds=cfg.FL_ROUNDS,
                                      local_epochs=cfg.FL_LOCAL_EPOCHS,
                                      lr=cfg.FL_LR)  # now internally decayed per round
        seed_f1s.append(fl_f1)
        seed_accs.append(fl_acc)
        print(f"  seed={seed}: F1={fl_f1:.4f}  Acc={fl_acc:.4f}")

    print("\n=== alpha_modal=3 :: WITH round-wise LR decay ===")
    print(f"  per-seed F1:  {[round(x,4) for x in seed_f1s]}")
    print(f"  mean F1={np.mean(seed_f1s):.4f}  std={np.std(seed_f1s):.4f}")
    print(f"  mean Acc={np.mean(seed_accs):.4f}  std={np.std(seed_accs):.4f}")

    print("\n=== alpha_modal=3 :: BASELINE (no decay, from your checkpoint) ===")
    print(f"  per-seed F1:  {[round(x,4) for x in BASELINE_ALPHA3['f1']]}")
    print(f"  mean F1={np.mean(BASELINE_ALPHA3['f1']):.4f}  "
          f"std={np.std(BASELINE_ALPHA3['f1']):.4f}")

    with open(args.out, "wb") as f:
        pickle.dump({
            "alpha_modal": 3,
            "seeds": cfg.SEEDS,
            "f1": seed_f1s,
            "acc": seed_accs,
            "f1_mean": float(np.mean(seed_f1s)),
            "f1_std": float(np.std(seed_f1s)),
            "acc_mean": float(np.mean(seed_accs)),
            "acc_std": float(np.std(seed_accs)),
        }, f)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()