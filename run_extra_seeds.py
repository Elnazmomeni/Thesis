import pickle
import numpy as np
import torch
from tqdm import tqdm
import argparse

from train_ravdess import (
    build_config, load_ravdess, build_client_datasets, train_fedavg,
    make_tensor_dataset, _fedartml_safe_seed,
)
from torch.utils.data import DataLoader

args = argparse.Namespace(
    ravdess_path="./RAVDESS",
    cache_path="./ravdess_features",
    images_dir="./images_ravdess_extra_seeds",
    device=None,
)
cfg = build_config(args)  # picks up NUM_CLIENTS, ALPHA_LABEL_FIXED, ALPHA_MODAL_SWEEP etc. as already set in train_ravdess.py

img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te = load_ravdess(cfg.RAVDESS_PATH, cfg.CACHE_PATH, cfg)

test_loader = DataLoader(make_tensor_dataset(img_te, aud_te, lbl_te),
                          batch_size=8, shuffle=False, num_workers=0)  # match run_alpha_sweep_full exactly

NEW_SEEDS = [7, 99, 2024, 31415, 8675309]
cfg.SEEDS = NEW_SEEDS

results_by_alpha_new = {}
for alpha_modal in cfg.ALPHA_MODAL_SWEEP:
    seed_f1s, seed_accs = [], []
    seed_modal_jsds, seed_modal_hds = [], []
    seed_label_jsds, seed_label_hds = [], []

    for seed in tqdm(cfg.SEEDS, desc=f"    alpha={alpha_modal}"):
        torch.manual_seed(seed)
        np.random.seed(seed)
        (client_ds, modal_jsd, modal_hd, label_jsd, label_hd, _) = build_client_datasets(
            img_tr, aud_tr, lbl_tr, alpha_modal=alpha_modal, cfg=cfg,
            alpha_label=cfg.ALPHA_LABEL_FIXED, num_clients=cfg.NUM_CLIENTS,
            random_state=_fedartml_safe_seed(seed, cfg))
        fl_f1, fl_acc = train_fedavg(client_ds, test_loader, cfg,
                                      fl_rounds=cfg.FL_ROUNDS, local_epochs=cfg.FL_LOCAL_EPOCHS,
                                      lr=cfg.FL_LR)
        seed_f1s.append(fl_f1); seed_accs.append(fl_acc)
        seed_modal_jsds.append(modal_jsd); seed_modal_hds.append(modal_hd)
        seed_label_jsds.append(label_jsd); seed_label_hds.append(label_hd)

    results_by_alpha_new[alpha_modal] = {
        "alpha_modal": alpha_modal, "alpha_label": cfg.ALPHA_LABEL_FIXED, "seeds": NEW_SEEDS,
        "f1": seed_f1s, "acc": seed_accs, "modal_jsd": seed_modal_jsds, "modal_hd": seed_modal_hds,
        "label_jsd": seed_label_jsds, "label_hd": seed_label_hds,
    }
    with open("ravdess_alpha_sweep_newseeds.pkl", "wb") as f:
        pickle.dump({"results_by_alpha": results_by_alpha_new, "done": set(results_by_alpha_new.keys())}, f)
    print(f"  [checkpoint saved after alpha={alpha_modal}]")

print("Done.")