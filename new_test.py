import pickle, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from ravdess_test import (
    build_config, load_ravdess, build_client_datasets, train_fedavg,
    make_tensor_dataset, _fedartml_safe_seed,
)

args = argparse.Namespace(ravdess_path="./RAVDESS", cache_path="./ravdess_features",
                          images_dir="./images_ravdess_diag", device=None)
cfg = build_config(args)
img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te = load_ravdess(cfg.RAVDESS_PATH, cfg.CACHE_PATH, cfg)
test_loader = DataLoader(make_tensor_dataset(img_te, aud_te, lbl_te),
                         batch_size=8, shuffle=False, num_workers=0)

out = {}
for k in [4, 10]:
    for seed in [13, 37, 256]:
        torch.manual_seed(seed); np.random.seed(seed)
        client_ds, *_ = build_client_datasets(
            img_tr, aud_tr, lbl_tr, alpha_modal=1000, cfg=cfg,
            alpha_label=cfg.ALPHA_LABEL_FIXED, num_clients=k,
            random_state=_fedartml_safe_seed(seed, cfg))
        (f1, acc), curve = train_fedavg(client_ds, test_loader, cfg, fl_rounds=300,
                                        local_epochs=1, lr=cfg.FL_LR, eval_every=10)
        out[(k, seed)] = {"final_f1": f1, "curve": curve}
        print(f"k={k} seed={seed} final F1={f1:.3f}")
        with open("diag_128.pkl", "wb") as f:
            pickle.dump(out, f)