"""
sanity_check_cl.py — explains WHY CL F1(macro) and Accuracy come out so
close, instead of just noting that they do. Retrains the centralised (CL)
model once using the exact same recipe as Step 2 in train_cremad.py, then
prints:
  - class distribution in the test set (to check it's actually balanced)
  - a full per-class precision/recall/F1 report
  - a confusion matrix

If per-class F1 scores are all roughly similar to each other, and the test
set class sizes are roughly similar to each other, that's your answer:
macro-F1 ≈ accuracy because there's no class the model is quietly failing
on that a macro average would otherwise expose.

Usage:
    python sanity_check_cl.py --cremad-path ./CREMA-D
"""

import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader

from train_cremad import (
    build_config, load_cremad, make_model, make_tensor_dataset, train_one_epoch,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad-path", default="./CREMA-D")
    ap.add_argument("--cache-path", default="./cremad_features")
    ap.add_argument("--images-dir", default="./images_cremad_v6")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = build_config(args)

    torch.manual_seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te = load_cremad(
        cfg.CREMAD_PATH, cfg.CACHE_PATH, cfg)

    print("\nClass distribution in TEST set:")
    for i, name in enumerate(cfg.EMOTION_NAMES):
        n = int(np.sum(lbl_te == i))
        print(f"  {name:10s}: {n:4d}  ({n/len(lbl_te):.1%})")
    print(f"  TOTAL test samples: {len(lbl_te)}")

    train_loader = DataLoader(make_tensor_dataset(img_tr, aud_tr, lbl_tr),
                               batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(make_tensor_dataset(img_te, aud_te, lbl_te),
                              batch_size=cfg.BATCH_SIZE, shuffle=False)

    model = make_model(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.CL_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.CL_EQUIV_EPOCHS, eta_min=1e-5)

    print(f"\nTraining CL for {cfg.CL_EQUIV_EPOCHS} epochs (same recipe as Step 2) ...")
    for epoch in range(cfg.CL_EQUIV_EPOCHS):
        train_one_epoch(model, train_loader, optimizer, cfg)
        scheduler.step()

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, aud, lbl in test_loader:
            preds = model(img.to(cfg.DEVICE), aud.to(cfg.DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nCL F1(macro)={f1:.4f}   Accuracy={acc:.4f}   (gap = {abs(f1-acc):.4f})")

    print("\nPer-class report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=cfg.EMOTION_NAMES, digits=3, zero_division=0))

    print("Confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(all_labels, all_preds)
    header = "            " + " ".join(f"{n[:4]:>6s}" for n in cfg.EMOTION_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{cfg.EMOTION_NAMES[i]:10s}  " + " ".join(f"{v:6d}" for v in row))


if __name__ == "__main__":
    main()