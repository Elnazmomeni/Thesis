"""
train_ravdess.py — RAVDESS multimodal FL pipeline, converted from your
notebook the same way train_cremad.py was, with the same fixes applied:

  1. BatchNorm -> GroupNorm in both branches (fixes the degenerate-batch /
     corrupted-running-stats issue we diagnosed on CREMA-D's client=4
     collapse — same architecture, same risk here).
  2. Alpha search range widened + split dense/sparse (0.01-10 dense,
     10-1000 sparse) so near-IID targets are reachable, without losing
     resolution where your existing targets already live.
  3. Removed the "if img.size(0) < 2: continue" guard — GroupNorm doesn't
     need it, so every sample gets used now.
  4. Same fedartml_local module layout, argparse config, and patch2-style
     checkpointed client sweep as train_cremad.py.
  5. Step 3 (alpha sweep) kept STANDALONE (independent training run) per
     your preference — not derived from Step 4's data — so it remains a
     genuine cross-check for Figure D.

Differences from train_cremad.py that are RAVDESS-specific, not fixes:
  - 8 emotion classes instead of 6.
  - Single video file provides BOTH audio and image features (RAVDESS
    "Video_Speech" files contain audio+video in one .mp4; CREMA-D has
    separate audio/video files).
  - Default FIXED_JSD_LEVELS/FIXED_HD_LEVELS match your notebook's values.

Usage:
    python train_ravdess.py --ravdess-path ./RAVDESS --images-dir ./images_ravdess

Recommended on a remote server:
    tmux new -s ravdess
    python train_ravdess.py --ravdess-path ./RAVDESS
    # Ctrl+b, d to detach; `tmux attach -t ravdess` to check back in later
"""

import argparse
import copy
import gc
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────
# numpy seed overflow guard (unchanged from CREMA-D)
# ─────────────────────────────────────────────────────────────────────────
import numpy as _np
import numbers

_NUMPY_SEED_MAX = 2**32 - 1

if not getattr(_np.random, "_seed_guard_installed", False):
    _original_np_seed = _np.random.seed

    def _safe_np_seed(seed=None):
        if isinstance(seed, numbers.Integral):
            seed = int(seed) % (_NUMPY_SEED_MAX + 1)
        return _original_np_seed(seed)

    _np.random.seed = _safe_np_seed

    _original_default_rng = _np.random.default_rng

    def _safe_default_rng(seed=None):
        if isinstance(seed, numbers.Integral):
            seed = int(seed) % (_NUMPY_SEED_MAX + 1)
        return _original_default_rng(seed)

    _np.random.default_rng = _safe_default_rng
    _np.random._seed_guard_installed = True
    print("  [guard] np.random.seed / default_rng patched — overflow-proof")


# ─────────────────────────────────────────────────────────────────────────
# Local FedArtML modules — same fedartml_local/ folder as train_cremad.py,
# reuse it rather than duplicating (these files aren't dataset-specific)
# ─────────────────────────────────────────────────────────────────────────
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fedartml_local"))
from fl_modality_heterogeneity import ModalityHeterogeneity
from fedartml_patch import SplitAsFederatedData


# ═════════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════════
def build_config(args):
    cfg = argparse.Namespace()
    cfg.RAVDESS_PATH = args.ravdess_path
    cfg.CACHE_PATH = args.cache_path
    cfg.IMAGES_DIR = args.images_dir

    cfg.RANDOM_STATE = 42
    cfg.SEEDS = [1, 42, 123, 512, 999]
    cfg.NUM_CLASSES = 8
    cfg.BATCH_SIZE = 64

    cfg.IMG_SIZE = 64
    cfg.IMG_DIM = cfg.IMG_SIZE * cfg.IMG_SIZE * 3
    cfg.N_MFCC = 40
    cfg.AUD_DIM = cfg.N_MFCC * 2

    if args.device is not None:
        cfg.DEVICE = args.device
        if cfg.DEVICE.startswith("cuda") and not torch.cuda.is_available():
            sys.exit(f"--device {cfg.DEVICE} was requested but CUDA isn't available on this machine.")
    else:
        cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.NUM_CLIENTS = 6
    cfg.FL_ROUNDS = 30
    cfg.FL_LOCAL_EPOCHS = 1
    cfg.FL_LR = 5e-4

    cfg.CL_EQUIV_EPOCHS = cfg.FL_ROUNDS * cfg.FL_LOCAL_EPOCHS  # 90
    cfg.CL_LR = 1e-3

    cfg.ALPHA_LABEL_FIXED = 1000

    cfg.ALPHA_SWEEP = [0.01, 0.08, 0.4, 1.0, 15, 1000]
    cfg.ALPHA_MODAL_SWEEP = [0.01, 0.08, 0.4, 1.0, 3, 15, 1000]
    cfg.CLIENT_SWEEP = [4, 6, 10, 20, 100]
    cfg.FL_ROUNDS_CLIENTS = 30

    # Same as your notebook's 4 levels, PLUS a near-IID anchor point at the
    # low end (same reasoning as CREMA-D's added 0.02/0.05: your alpha
    # search now reaches up to alpha=1000, so it can actually hit a
    # near-zero target). RAVDESS's original levels already start lower
    # than CREMA-D's did (0.05 vs 0.10 for JSD, 0.10 vs 0.16 for HD), so
    # the new anchor is scaled down accordingly rather than reusing
    # CREMA-D's exact 0.02/0.05 values.
    cfg.FIXED_JSD_LEVELS = [0.01, 0.05, 0.15, 0.30, 0.45]
    cfg.FIXED_HD_LEVELS = [0.02, 0.10, 0.25, 0.45, 0.70]

    cfg.MAX_PER_CLASS_ALPHA_SWEEP = 20

    cfg.EMOTION_MAP = {
        "01": 0, "02": 1, "03": 2, "04": 3,
        "05": 4, "06": 5, "07": 6, "08": 7,
    }
    cfg.EMOTION_NAMES = ["neutral", "calm", "happy", "sad",
                          "angry", "fearful", "disgust", "surprised"]

    os.makedirs(cfg.IMAGES_DIR, exist_ok=True)

    cfg.FEDARTML_SEED_CAP = 100_000
    return cfg


def _fedartml_safe_seed(random_state: int, cfg) -> int:
    return int(random_state) % cfg.FEDARTML_SEED_CAP


# ═════════════════════════════════════════════════════════════════════════
# Feature extraction — RAVDESS-specific: audio AND image both come from
# the SAME .mp4 file (unlike CREMA-D's separate AudioWAV/VideoFlash dirs)
# ═════════════════════════════════════════════════════════════════════════
def extract_audio_features(vid_path, cfg):
    import librosa
    y, sr = librosa.load(vid_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.N_MFCC)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(np.float32)


def extract_image_feature(vid_path, cfg):
    import cv2
    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return np.zeros(cfg.IMG_DIM, dtype=np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    return frame.astype(np.float32).ravel() / 255.0


def extract_and_cache_features(ravdess_path, cache_path, cfg):
    os.makedirs(cache_path, exist_ok=True)
    img_cache = os.path.join(cache_path, "img.npy")
    aud_cache = os.path.join(cache_path, "aud.npy")
    lbl_cache = os.path.join(cache_path, "labels.npy")

    if all(os.path.exists(p) for p in [img_cache, aud_cache, lbl_cache]):
        print("  Loading from cache ...")
        img = np.load(img_cache)
        aud = np.load(aud_cache)
        lbl = np.load(lbl_cache)
        print(f"  img={img.shape}  aud={aud.shape}  lbl={lbl.shape}")
        return img, aud, lbl

    video_paths = []
    for d in sorted(os.listdir(ravdess_path)):
        actor_dir = os.path.join(ravdess_path, d)
        if not os.path.isdir(actor_dir):
            continue
        for f in sorted(os.listdir(actor_dir)):
            if f.lower().endswith((".mp4", ".flv", ".avi")):
                video_paths.append(os.path.join(actor_dir, f))
    print(f"  Found {len(video_paths)} video files")

    imgs, auds, lbls, skipped = [], [], [], 0
    for vid_path in tqdm(video_paths, desc="  Extracting features"):
        base = os.path.basename(vid_path)
        stem = os.path.splitext(base)[0]
        fields = stem.split("-")
        if len(fields) < 7:
            skipped += 1
            continue
        emotion_code = fields[2]
        if emotion_code not in cfg.EMOTION_MAP:
            skipped += 1
            continue
        label = cfg.EMOTION_MAP[emotion_code]

        try:
            aud_feat = extract_audio_features(vid_path, cfg)
        except Exception:
            skipped += 1
            continue

        try:
            img_feat = extract_image_feature(vid_path, cfg)
        except Exception:
            img_feat = None
        if img_feat is None:
            img_feat = np.zeros(cfg.IMG_DIM, dtype=np.float32)

        imgs.append(img_feat)
        auds.append(aud_feat)
        lbls.append(label)

    if skipped:
        print(f"  Skipped {skipped} files")
    img_arr = np.stack(imgs).astype(np.float32)
    aud_arr = np.stack(auds).astype(np.float32)
    lbl_arr = np.array(lbls, dtype=np.int64)
    np.save(img_cache, img_arr)
    np.save(aud_cache, aud_arr)
    np.save(lbl_cache, lbl_arr)
    return img_arr, aud_arr, lbl_arr


def load_ravdess(ravdess_path, cache_path, cfg, test_ratio=0.2):
    print(f"  Loading RAVDESS from: {ravdess_path}")
    img_all, aud_all, lbl_all = extract_and_cache_features(ravdess_path, cache_path, cfg)
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    tr_idx, te_idx = [], []
    for c in range(cfg.NUM_CLASSES):
        c_idx = np.where(lbl_all == c)[0]
        rng.shuffle(c_idx)
        n_te = max(1, int(len(c_idx) * test_ratio))
        te_idx.extend(c_idx[:n_te])
        tr_idx.extend(c_idx[n_te:])
    tr_idx = np.array(tr_idx)
    te_idx = np.array(te_idx)
    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    img_tr, img_te = img_all[tr_idx], img_all[te_idx]
    aud_tr, aud_te = aud_all[tr_idx], aud_all[te_idx]
    lbl_tr, lbl_te = lbl_all[tr_idx], lbl_all[te_idx]
    aud_mean = aud_tr.mean(axis=0, keepdims=True)
    aud_std = aud_tr.std(axis=0, keepdims=True) + 1e-8
    aud_tr = (aud_tr - aud_mean) / aud_std
    aud_te = (aud_te - aud_mean) / aud_std
    print(f"  train={img_tr.shape[0]}  test={img_te.shape[0]}")
    return img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te


# ═════════════════════════════════════════════════════════════════════════
# Model architecture — FIX 1: BatchNorm -> GroupNorm
# ═════════════════════════════════════════════════════════════════════════
class ImageBranch(nn.Module):
    def __init__(self, cfg, embed_dim=128, use_batchnorm=False):
        super().__init__()
 
        def norm(num_groups, num_channels):
            if use_batchnorm:
                return nn.BatchNorm2d(num_channels)
            return nn.GroupNorm(num_groups, num_channels)
 
        self.net = nn.Sequential(
            nn.Unflatten(1, (3, cfg.IMG_SIZE, cfg.IMG_SIZE)),
            nn.Conv2d(3, 32, 3, padding=1), norm(8, 32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), norm(8, 64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), norm(8, 128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8192, embed_dim), nn.ReLU(),
        )
 
    def forward(self, x):
        return self.net(x)
 
 

class AudioBranch(nn.Module):
    def __init__(self, input_dim, embed_dim=128, use_batchnorm=False):
        super().__init__()
 
        def norm(num_groups, num_channels):
            if use_batchnorm:
                return nn.BatchNorm1d(num_channels)
            return nn.GroupNorm(num_groups, num_channels)
 
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), norm(16, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), norm(16, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, embed_dim), nn.ReLU(),
        )
 
    def forward(self, x):
        return self.net(x)
 
class MultimodalNet(nn.Module):
    def __init__(self, cfg, num_classes, embed_dim=128, aud_input_dim=None,
                 use_batchnorm=False):
        super().__init__()
        aud_input_dim = aud_input_dim or cfg.AUD_DIM
        self.img_branch = ImageBranch(cfg, embed_dim, use_batchnorm=use_batchnorm)
        self.aud_branch = AudioBranch(aud_input_dim, embed_dim, use_batchnorm=use_batchnorm)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
 
    def forward(self, img, aud):
        return self.classifier(torch.cat([self.img_branch(img), self.aud_branch(aud)], dim=1))
 
 
def make_model(cfg, use_batchnorm=False):
    """
    use_batchnorm=True  -> CL baseline (full, stable batch sizes; BatchNorm
                            gives its normal, usually-better performance)
    use_batchnorm=False -> everything federated (client models can see
                            tiny/degenerate batches; GroupNorm avoids the
                            running-stats corruption seen in the CREMA-D
                            client=4 collapse)
    """
    return MultimodalNet(cfg, num_classes=cfg.NUM_CLASSES, embed_dim=128,
                          aud_input_dim=cfg.AUD_DIM,
                          use_batchnorm=use_batchnorm).to(cfg.DEVICE)


def make_tensor_dataset(img, aud, labels):
    return TensorDataset(
        torch.from_numpy(img).float(),
        torch.from_numpy(aud).float(),
        torch.from_numpy(labels).long(),
    )


# ═════════════════════════════════════════════════════════════════════════
# Client partitioning (unchanged logic from CREMA-D — dataset-agnostic)
# ═════════════════════════════════════════════════════════════════════════
def partition_data_fedartml(img_tr, aud_tr, lbl_tr, num_clients, alpha_label, cfg, random_state=42):
    fedartml_seed = _fedartml_safe_seed(random_state, cfg)
    federater = SplitAsFederatedData(random_state=fedartml_seed)
    dummy_images = np.arange(len(lbl_tr)).reshape(-1, 1).astype(np.float32)
    fed_data, ids_list, num_missing, distances = federater.create_clients(
        image_list=dummy_images,
        label_list=lbl_tr,
        num_clients=num_clients,
        prefix_cli="client",
        method="dirichlet",
        alpha=alpha_label,
        sigma_noise=0,
        feat_skew_method="gaussian-noise",
        quant_skew_method="no-quant-skew",
        spa_temp_skew_method="no-spatemp-skew",
    )
    idx_per_client = ids_list["without_class_completion"]
    partitions = []
    for c_idx_list in idx_per_client:
        idx = np.array(c_idx_list, dtype=int)
        if len(idx) == 0:
            idx = np.array([0])
        partitions.append((img_tr[idx], aud_tr[idx], lbl_tr[idx]))
    label_jsd = distances["without_class_completion"]["jensen-shannon"]
    label_hd = distances["without_class_completion"]["hellinger"]
    return partitions, label_jsd, label_hd


def build_client_datasets(img_tr, aud_tr, lbl_tr, alpha_modal, cfg,
                           alpha_label=None, num_clients=None, random_state=42):
    if alpha_label is None:
        alpha_label = cfg.ALPHA_LABEL_FIXED
    if num_clients is None:
        num_clients = cfg.NUM_CLIENTS
    partitions, label_jsd, label_hd = partition_data_fedartml(
        img_tr, aud_tr, lbl_tr, num_clients=num_clients,
        alpha_label=alpha_label, cfg=cfg, random_state=random_state)

    mh = ModalityHeterogeneity(modality_names=["image", "audio"], random_state=random_state)

    all_img = np.concatenate([p[0] for p in partitions], axis=0)
    all_aud = np.concatenate([p[1] for p in partitions], axis=0)
    all_lbl = np.concatenate([p[2] for p in partitions], axis=0)

    split_sizes = [len(p[2]) for p in partitions]
    split_ends = np.cumsum(split_sizes)
    split_starts = np.concatenate([[0], split_ends[:-1]])

    modality_arrays = {"image": all_img, "audio": all_aud}

    joint_data, _ = mh.assign_modalities_to_clients(
        modality_arrays=modality_arrays, y=all_lbl, num_clients=num_clients,
        alpha=alpha_modal, prefix_cli="client")

    del partitions, all_img, all_aud, all_lbl
    gc.collect()

    client_names = [f"client_{i+1}" for i in range(num_clients)]
    client_order = client_names[:]
    client_data_combined = {}
    client_datasets = []

    for c_idx, c_name in enumerate(client_names):
        s = split_starts[c_idx]
        e = split_ends[c_idx]
        c_entry = {
            "y": joint_data[c_name]["y"][s:e],
            "masks": joint_data[c_name]["masks"][s:e],
            "mask_ids": joint_data[c_name]["mask_ids"][s:e],
            "image": joint_data[c_name]["image"][s:e],
            "audio": joint_data[c_name]["audio"][s:e],
        }
        client_data_combined[c_name] = c_entry
        client_datasets.append(TensorDataset(
            torch.from_numpy(c_entry["image"]).float(),
            torch.from_numpy(c_entry["audio"]).float(),
            torch.from_numpy(c_entry["y"]).long(),
        ))

    del joint_data
    gc.collect()

    scores = mh.compute_modality_heterogeneity_score(client_data_combined)
    modal_jsd = scores["mean_js_divergence"]
    modal_hd = scores["mean_hellinger"]

    del client_data_combined
    gc.collect()

    return client_datasets, modal_jsd, modal_hd, label_jsd, label_hd, client_order


# ═════════════════════════════════════════════════════════════════════════
# Alpha search — FIX 2: widened, split dense/sparse range
# ═════════════════════════════════════════════════════════════════════════
_ALPHA_CACHE = {}


def _find_alpha(kind, target, num_clients, img_tr, aud_tr, lbl_tr, cfg, probe_seeds=None):
    if probe_seeds is None:
        probe_seeds = cfg.SEEDS

    key = (kind, round(float(target), 4), num_clients)
    if key in _ALPHA_CACHE:
        return _ALPHA_CACHE[key]

    # Dense in 0.01-10 (where your real targets live), sparse in 10-1000
    # (reach for a near-IID target if you add one, without diluting
    # resolution where it matters). See train_cremad.py discussion for why.
    candidates = np.concatenate([
        np.logspace(-2, 1, 50),
        np.logspace(1, 3, 21)[1:],
    ])
    n_candidates = len(candidates)
    best_alpha, best_diff = candidates[0], 1e9   # <-- must be set before the loop

    print(f"    [{kind.upper()} search] target={target:.4f}  num_clients={num_clients}  "
          f"({n_candidates} candidates, averaged over {len(probe_seeds)} seeds: {probe_seeds})")

    for i, a in enumerate(candidates):
        vals = []
        for ps in probe_seeds:
            _, jsd, hd, _, _, _ = build_client_datasets(
                img_tr, aud_tr, lbl_tr, alpha_modal=a, cfg=cfg,
                alpha_label=cfg.ALPHA_LABEL_FIXED, num_clients=num_clients,
                random_state=_fedartml_safe_seed(ps, cfg))
            vals.append(jsd if kind == "jsd" else hd)
            del jsd, hd
        gc.collect()
        if cfg.DEVICE == "cuda":
            torch.cuda.empty_cache()

        mean_val = float(np.mean(vals))
        spread = float(np.ptp(vals))
        diff = abs(mean_val - target)
        if i % 20 == 0:
            print(f"      {i}/{n_candidates}  alpha={a:.4f}  mean_achieved={mean_val:.4f}  "
                  f"spread_across_seeds={spread:.4f}  best_alpha={best_alpha:.4f}  best_diff={best_diff:.4f}")
        if diff < best_diff:
            best_diff, best_alpha = diff, a

    _ALPHA_CACHE[key] = best_alpha
    return best_alpha


def find_alpha_for_jsd(target_jsd, num_clients, random_state, cfg, img_tr, aud_tr, lbl_tr):
    return _find_alpha("jsd", target_jsd, num_clients, img_tr, aud_tr, lbl_tr, cfg)


def find_alpha_for_hd(target_hd, num_clients, random_state, cfg, img_tr, aud_tr, lbl_tr):
    return _find_alpha("hd", target_hd, num_clients, img_tr, aud_tr, lbl_tr, cfg)


# ═════════════════════════════════════════════════════════════════════════
# Training / evaluation / FedAvg — FIX 3: removed batch-size-1 guard
# (GroupNorm doesn't need it, unlike BatchNorm)
# ═════════════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, cfg):
    model.train()
    for img, aud, lbl in loader:
        img, aud, lbl = img.to(cfg.DEVICE), aud.to(cfg.DEVICE), lbl.to(cfg.DEVICE)
        optimizer.zero_grad()
        criterion(model(img, aud), lbl).backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    all_preds, all_labels = [], []
    for img, aud, lbl in loader:
        preds = model(img.to(cfg.DEVICE), aud.to(cfg.DEVICE)).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbl.numpy())
    f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    acc = float(accuracy_score(all_labels, all_preds))
    return f1, acc


def train_centralised(img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cfg):
    print("\n" + "═" * 60)
    print("STEP 2 — Centralised (CL) training  [compute-equalised]")
    print(f"  CL epochs = FL_ROUNDS × FL_LOCAL_EPOCHS = "
          f"{cfg.FL_ROUNDS} × {cfg.FL_LOCAL_EPOCHS} = {cfg.CL_EQUIV_EPOCHS}")
    print("═" * 60)

    train_loader = DataLoader(make_tensor_dataset(img_tr, aud_tr, lbl_tr),
                               batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0,
                               pin_memory=(cfg.DEVICE == "cuda"))
    test_loader = DataLoader(make_tensor_dataset(img_te, aud_te, lbl_te),
                              batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0,
                              pin_memory=(cfg.DEVICE == "cuda"))

    model = make_model(cfg, use_batchnorm=True)
    optimizer = optim.Adam(model.parameters(), lr=cfg.CL_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.CL_EQUIV_EPOCHS, eta_min=1e-5)

    for epoch in tqdm(range(cfg.CL_EQUIV_EPOCHS), desc="  CL epochs"):
        train_one_epoch(model, train_loader, optimizer, cfg)
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            f1, acc = evaluate(model, test_loader, cfg)
            print(f"  Epoch {epoch+1:3d}:  F1={f1:.4f}  Acc={acc:.4f}")

    cl_f1, cl_acc = evaluate(model, test_loader, cfg)
    print(f"\n  CL final:  F1={cl_f1:.4f}   Acc={cl_acc:.4f}")
    del model, optimizer, scheduler
    if cfg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return cl_f1, cl_acc


def _get_branch_keys(state_dict):
    img_keys = [k for k in state_dict if k.startswith("img_branch")]
    aud_keys = [k for k in state_dict if k.startswith("aud_branch")]
    clf_keys = [k for k in state_dict if k.startswith("classifier")]
    return img_keys, aud_keys, clf_keys


def train_fedavg(client_datasets, test_loader, cfg, fl_rounds=None, local_epochs=None, lr=None):
    if fl_rounds is None:
        fl_rounds = cfg.FL_ROUNDS
    if local_epochs is None:
        local_epochs = cfg.FL_LOCAL_EPOCHS
    if lr is None:
        lr = cfg.FL_LR

    global_model = make_model(cfg)
    local_model = make_model(cfg)

    sample_weights = np.array([len(ds) for ds in client_datasets], dtype=np.float64)
    sample_weights /= sample_weights.sum()

    MIN_CLIENT_SAMPLES = max(2, cfg.BATCH_SIZE // 4)
    any_round_trained = False

    for rnd in tqdm(range(fl_rounds), desc="      FedAvg rounds", leave=False):
        global_sd = global_model.state_dict()
        img_keys, aud_keys, clf_keys = _get_branch_keys(global_sd)
        all_keys = img_keys + aud_keys + clf_keys

        weighted_sum = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                         for k, v in global_sd.items() if k in all_keys}
        total_weight = 0.0
        any_client_trained_this_round = False

        for c_i, c_ds in enumerate(client_datasets):
            if len(c_ds) < MIN_CLIENT_SAMPLES:
                continue
            local_model.load_state_dict(copy.deepcopy(global_sd))
            loader = DataLoader(c_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                 num_workers=0, drop_last=False)
            if len(loader) == 0:
                continue
            opt = optim.Adam(local_model.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(local_epochs):
                train_one_epoch(local_model, loader, opt, cfg)

            w = sample_weights[c_i]
            local_sd = local_model.state_dict()
            for k in all_keys:
                weighted_sum[k] += w * local_sd[k].detach().cpu().float()
            total_weight += w
            any_client_trained_this_round = True
            del local_sd
            gc.collect()

        if not any_client_trained_this_round:
            continue

        any_round_trained = True
        new_state = {k: (v / total_weight) for k, v in weighted_sum.items()}
        global_model.load_state_dict({k: v.to(cfg.DEVICE) for k, v in new_state.items()})
        del weighted_sum
        gc.collect()

    if not any_round_trained:
        print(f"    [WARNING] train_fedavg: no client ever had >= {MIN_CLIENT_SAMPLES} "
              "samples in any round — returning untrained global model performance.")

    result = evaluate(global_model, test_loader, cfg)
    del global_model, local_model
    if cfg.DEVICE == "cuda":
        torch.cuda.empty_cache()
    return result


# ═════════════════════════════════════════════════════════════════════════
# Step 3 — standalone alpha sweep (kept independent of Step 4, per your
# preference, so it stays a real cross-check on Figure D rather than the
# same data replotted)
# ═════════════════════════════════════════════════════════════════════════
def run_alpha_sweep_full(img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cl_f1, cl_acc, cfg,
                          checkpoint_path="./checkpoints/ravdess_alpha_sweep_ckpt.pkl"):
    print("\n" + "═" * 60)
    print("STEP 3 — Alpha-modal sweep [FULL dataset, checkpointed]")
    print(f"  alphas = {cfg.ALPHA_MODAL_SWEEP}   num_clients = {cfg.NUM_CLIENTS}   "
          f"seeds = {cfg.SEEDS}")
    print("═" * 60)
 
    test_loader = DataLoader(
        make_tensor_dataset(img_te, aud_te, lbl_te),
        batch_size=8, shuffle=False, num_workers=0)
 
    # ── try to resume ────────────────────────────────────────────────
    results_by_alpha = {}
    done = set()
    try:
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        results_by_alpha = ckpt.get("results_by_alpha", {})
        done = set(ckpt.get("done", set()))
        print(f"  [resume] loaded checkpoint, {len(done)} alpha(s) already complete: "
              f"{sorted(done)}")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("  [resume] no usable checkpoint found — starting fresh")
 
    def _checkpoint():
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"results_by_alpha": results_by_alpha, "done": done}, f)
 
    for i, alpha_modal in enumerate(cfg.ALPHA_MODAL_SWEEP):
        alpha_key = round(float(alpha_modal), 6)
        if alpha_key in done:
            print(f"\n  [skip] alpha_modal={alpha_modal} already in checkpoint")
            continue
 
        print(f"\n  alpha_modal = {alpha_modal}   [{i+1}/{len(cfg.ALPHA_MODAL_SWEEP)}]")
        seed_f1s, seed_accs = [], []
        seed_modal_jsds, seed_modal_hds = [], []
        seed_label_jsds, seed_label_hds = [], []
 
        for seed in tqdm(cfg.SEEDS, desc=f"    alpha={alpha_modal}"):
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
                                          lr=cfg.FL_LR)
 
            seed_f1s.append(fl_f1)
            seed_accs.append(fl_acc)
            seed_modal_jsds.append(modal_jsd)
            seed_modal_hds.append(modal_hd)
            seed_label_jsds.append(label_jsd)
            seed_label_hds.append(label_hd)
 
            del client_ds
            gc.collect()
            if cfg.DEVICE == "cuda":
                torch.cuda.empty_cache()
 
        print(f"    Mean F1={np.mean(seed_f1s):.4f}  Acc={np.mean(seed_accs):.4f}  "
              f"modal_JSD={np.mean(seed_modal_jsds):.4f}  modal_HD={np.mean(seed_modal_hds):.4f}")
 
        results_by_alpha[alpha_key] = {
            "alpha_modal": alpha_modal,
            "alpha_label": cfg.ALPHA_LABEL_FIXED,
            "seeds": cfg.SEEDS,
            "f1": seed_f1s, "acc": seed_accs,
            "modal_jsd": seed_modal_jsds, "modal_hd": seed_modal_hds,
            "label_jsd": seed_label_jsds, "label_hd": seed_label_hds,
            "modal_jsd_mean": float(np.mean(seed_modal_jsds)),
            "modal_jsd_std": float(np.std(seed_modal_jsds)),
            "modal_hd_mean": float(np.mean(seed_modal_hds)),
            "modal_hd_std": float(np.std(seed_modal_hds)),
            "label_jsd_mean": float(np.mean(seed_label_jsds)),
            "label_jsd_std": float(np.std(seed_label_jsds)),
            "label_hd_mean": float(np.mean(seed_label_hds)),
            "label_hd_std": float(np.std(seed_label_hds)),
            "f1_mean": float(np.mean(seed_f1s)),
            "f1_std": float(np.std(seed_f1s)),
            "acc_mean": float(np.mean(seed_accs)),
            "acc_std": float(np.std(seed_accs)),
        }
        done.add(alpha_key)
        _checkpoint()
        print(f"  [checkpoint] saved after alpha={alpha_modal}")
 
    # Return in the same order as cfg.ALPHA_MODAL_SWEEP, list-of-dicts,
    # exactly matching your original function's output format — nothing
    # downstream (plot_alpha_sweep_figure, save_results) needs to change.
    all_results = [results_by_alpha[round(float(a), 6)] for a in cfg.ALPHA_MODAL_SWEEP]
    return all_results
 
# ═════════════════════════════════════════════════════════════════════════
# Step 4 — client sweep (checkpointed, patch2-style resume)
# ═════════════════════════════════════════════════════════════════════════
def run_client_sweep(img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cl_f1, cl_acc,
                      fixed_jsd_levels, fixed_hd_levels, cfg, checkpoint_path):
    assert fixed_jsd_levels is not None and fixed_hd_levels is not None, \
        "Set FIXED_JSD_LEVELS / FIXED_HD_LEVELS before running Step 4."

    print("\n" + "═" * 60)
    print("STEP 4 — Client sweep [patch2: incremental client_counts/CL_*]")
    print(f"  alpha_label FIXED = {cfg.ALPHA_LABEL_FIXED}  (FedArtML)")
    print(f"  JSD levels = {fixed_jsd_levels}")
    print(f"  HD levels  = {fixed_hd_levels}")
    print("  Training on FULL dataset per client sweep point.")
    print("═" * 60)

    test_loader = DataLoader(make_tensor_dataset(img_te, aud_te, lbl_te),
                              batch_size=8, shuffle=False, num_workers=0)

    results_jsd, results_hd = None, None
    done = set()
    try:
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        results_jsd = ckpt.get("results_jsd")
        results_hd = ckpt.get("results_hd")
        done = set(ckpt.get("done", set()))
        print(f"  [resume] loaded checkpoint, {len(done)} (client,type,level) combos already complete")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        print("  [resume] no usable checkpoint found — starting fresh")

    if results_jsd is None:
        results_jsd = {"CL_f1_mean": [], "CL_acc_mean": [], "client_counts": [],
                        "alpha_label": cfg.ALPHA_LABEL_FIXED}
        for j in fixed_jsd_levels:
            results_jsd[f"FL_f1_mean_JSD_{j:.2f}"] = []
            results_jsd[f"FL_f1_std_JSD_{j:.2f}"] = []
            results_jsd[f"FL_acc_mean_JSD_{j:.2f}"] = []
            results_jsd[f"FL_acc_std_JSD_{j:.2f}"] = []
            results_jsd[f"achieved_jsd_mean_{j:.2f}"] = []
            results_jsd[f"achieved_jsd_std_{j:.2f}"] = []

    if results_hd is None:
        results_hd = {"CL_f1_mean": [], "CL_acc_mean": [], "client_counts": [],
                       "hd_levels_used": fixed_hd_levels, "alpha_label": cfg.ALPHA_LABEL_FIXED}
        for h in fixed_hd_levels:
            results_hd[f"FL_f1_mean_HD_{h:.2f}"] = []
            results_hd[f"FL_f1_std_HD_{h:.2f}"] = []
            results_hd[f"FL_acc_mean_HD_{h:.2f}"] = []
            results_hd[f"FL_acc_std_HD_{h:.2f}"] = []
            results_hd[f"achieved_hd_mean_{h:.2f}"] = []
            results_hd[f"achieved_hd_std_{h:.2f}"] = []

    def _checkpoint():
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"results_jsd": results_jsd, "results_hd": results_hd, "done": done}, f)

    run_idx = 0
    total_runs = (len(cfg.CLIENT_SWEEP) * (len(fixed_jsd_levels) + len(fixed_hd_levels)) * len(cfg.SEEDS))

    for n_clients in cfg.CLIENT_SWEEP:
        n_clients = max(2, n_clients)

        for jsd_level in fixed_jsd_levels:
            level_key = (n_clients, "jsd", round(jsd_level, 4))
            if level_key in done:
                print(f"  [skip] clients={n_clients} JSD={jsd_level:.2f} already in checkpoint")
                run_idx += len(cfg.SEEDS)
                continue

            seed_f1s, seed_accs, seed_act_jsds = [], [], []
            for seed in cfg.SEEDS:
                run_idx += 1
                torch.manual_seed(seed)
                np.random.seed(seed)

                alpha_modal = find_alpha_for_jsd(jsd_level, n_clients, seed, cfg,
                                                  img_tr=img_tr, aud_tr=aud_tr, lbl_tr=lbl_tr)

                (client_ds, act_jsd, act_hd, label_jsd, _, _) = build_client_datasets(
                    img_tr, aud_tr, lbl_tr, alpha_modal=alpha_modal, cfg=cfg,
                    alpha_label=cfg.ALPHA_LABEL_FIXED, num_clients=n_clients,
                    random_state=_fedartml_safe_seed(seed, cfg))

                fl_f1, fl_acc = train_fedavg(client_ds, test_loader, cfg,
                                              fl_rounds=cfg.FL_ROUNDS_CLIENTS,
                                              local_epochs=cfg.FL_LOCAL_EPOCHS, lr=cfg.FL_LR)

                seed_f1s.append(fl_f1)
                seed_accs.append(fl_acc)
                seed_act_jsds.append(act_jsd)
                print(f"  [{run_idx:3d}/{total_runs}]  clients={n_clients:<4}  JSD_tgt={jsd_level:.2f}  "
                      f"JSD_achieved={act_jsd:.4f}  label_JSD={label_jsd:.4f}  seed={seed}  "
                      f"F1={fl_f1:.4f} Acc={fl_acc:.4f}")

                del client_ds
                gc.collect()
                if cfg.DEVICE == "cuda":
                    torch.cuda.empty_cache()

            results_jsd[f"FL_f1_mean_JSD_{jsd_level:.2f}"].append(round(float(np.mean(seed_f1s)), 4))
            results_jsd[f"FL_f1_std_JSD_{jsd_level:.2f}"].append(round(float(np.std(seed_f1s)), 4))
            results_jsd[f"FL_acc_mean_JSD_{jsd_level:.2f}"].append(round(float(np.mean(seed_accs)), 4))
            results_jsd[f"FL_acc_std_JSD_{jsd_level:.2f}"].append(round(float(np.std(seed_accs)), 4))
            results_jsd[f"achieved_jsd_mean_{jsd_level:.2f}"].append(round(float(np.mean(seed_act_jsds)), 4))
            results_jsd[f"achieved_jsd_std_{jsd_level:.2f}"].append(round(float(np.std(seed_act_jsds)), 4))

            done.add(level_key)
            _checkpoint()
            print(f"  [checkpoint] saved after clients={n_clients} JSD={jsd_level:.2f}")

        for hd_level in fixed_hd_levels:
            level_key = (n_clients, "hd", round(hd_level, 4))
            if level_key in done:
                print(f"  [skip] clients={n_clients} HD={hd_level:.2f} already in checkpoint")
                run_idx += len(cfg.SEEDS)
                continue

            seed_f1s, seed_accs, seed_act_hds = [], [], []
            for seed in cfg.SEEDS:
                run_idx += 1
                torch.manual_seed(seed)
                np.random.seed(seed)

                alpha_modal = find_alpha_for_hd(hd_level, n_clients, seed, cfg,
                                                 img_tr=img_tr, aud_tr=aud_tr, lbl_tr=lbl_tr)

                (client_ds, act_jsd, act_hd, label_jsd, _, _) = build_client_datasets(
                    img_tr, aud_tr, lbl_tr, alpha_modal=alpha_modal, cfg=cfg,
                    alpha_label=cfg.ALPHA_LABEL_FIXED, num_clients=n_clients,
                    random_state=_fedartml_safe_seed(seed, cfg))

                fl_f1, fl_acc = train_fedavg(client_ds, test_loader, cfg,
                                              fl_rounds=cfg.FL_ROUNDS_CLIENTS,
                                              local_epochs=cfg.FL_LOCAL_EPOCHS, lr=cfg.FL_LR)

                seed_f1s.append(fl_f1)
                seed_accs.append(fl_acc)
                seed_act_hds.append(act_hd)
                print(f"  [{run_idx:3d}/{total_runs}]  clients={n_clients:<4}  HD_tgt={hd_level:.2f}  "
                      f"HD_achieved={act_hd:.4f}  label_JSD={label_jsd:.4f}  seed={seed}  "
                      f"F1={fl_f1:.4f} Acc={fl_acc:.4f}")

                del client_ds
                gc.collect()
                if cfg.DEVICE == "cuda":
                    torch.cuda.empty_cache()

            results_hd[f"FL_f1_mean_HD_{hd_level:.2f}"].append(round(float(np.mean(seed_f1s)), 4))
            results_hd[f"FL_f1_std_HD_{hd_level:.2f}"].append(round(float(np.std(seed_f1s)), 4))
            results_hd[f"FL_acc_mean_HD_{hd_level:.2f}"].append(round(float(np.mean(seed_accs)), 4))
            results_hd[f"FL_acc_std_HD_{hd_level:.2f}"].append(round(float(np.std(seed_accs)), 4))
            results_hd[f"achieved_hd_mean_{hd_level:.2f}"].append(round(float(np.mean(seed_act_hds)), 4))
            results_hd[f"achieved_hd_std_{hd_level:.2f}"].append(round(float(np.std(seed_act_hds)), 4))

            done.add(level_key)
            _checkpoint()
            print(f"  [checkpoint] saved after clients={n_clients} HD={hd_level:.2f}")

        if n_clients not in results_jsd["client_counts"]:
            results_jsd["client_counts"].append(n_clients)
            results_jsd["CL_f1_mean"].append(round(cl_f1, 4))
            results_jsd["CL_acc_mean"].append(round(cl_acc, 4))
        if n_clients not in results_hd["client_counts"]:
            results_hd["client_counts"].append(n_clients)
            results_hd["CL_f1_mean"].append(round(cl_f1, 4))
            results_hd["CL_acc_mean"].append(round(cl_acc, 4))
        _checkpoint()

        print(f"  [checkpoint] client_count={n_clients} fully complete "
              f"(client_counts now = {results_jsd['client_counts']})")

    return results_jsd, results_hd


# ═════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════
C_CL = "#D62728"
C_FL_F1 = "#1F77B4"
C_FL_ACC = "#FF7F0E"
JSD_PAL = ["#1A9850", "#91CF60", "#FEE08B", "#FC8D59"]
HD_PAL = ["#313695", "#4575B4", "#ABD9E9", "#F46D43"]

BASE_RC = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333", "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "sans-serif",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9,
}


def _savefig(fig, name, cfg):
    path = os.path.join(cfg.IMAGES_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")


def plot_alpha_sweep_figure(sweep_results, dist_key, dist_label, fig_letter, filename, cl_f1, cl_acc, cfg):
    plt.rcParams.update(BASE_RC)
    alphas = [r["alpha_modal"] for r in sweep_results]
    dist_vals = [r[dist_key] for r in sweep_results]
    f1_means = [r["f1_mean"] for r in sweep_results]
    f1_stds = [r["f1_std"] for r in sweep_results]
    acc_means = [r["acc_mean"] for r in sweep_results]
    acc_stds = [r["acc_std"] for r in sweep_results]

    order = np.argsort(dist_vals)
    alphas = [alphas[i] for i in order]
    dist_vals = [dist_vals[i] for i in order]
    f1_means = np.array([f1_means[i] for i in order])
    f1_stds = np.array([f1_stds[i] for i in order])
    acc_means = np.array([acc_means[i] for i in order])
    acc_stds = np.array([acc_stds[i] for i in order])
    x = np.array(dist_vals)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    plt.subplots_adjust(top=0.78, bottom=0.13, left=0.09, right=0.72)
    lw = 2.0
    ax.axhline(cl_f1, color=C_CL, lw=lw, label="CL — F1-Score")
    ax.axhline(cl_acc, color="#2CA02C", lw=lw, label="CL — Accuracy")
    ax.plot(x, f1_means, color=C_FL_F1, lw=lw, linestyle="--", marker="^", markersize=7,
            label="FL (FedAvg) — F1-Score")
    ax.fill_between(x, f1_means - f1_stds, f1_means + f1_stds, color=C_FL_F1, alpha=0.15)
    ax.plot(x, acc_means, color=C_FL_ACC, lw=lw, linestyle="--", marker="D", markersize=6,
            label="FL (FedAvg) — Accuracy")
    ax.fill_between(x, acc_means - acc_stds, acc_means + acc_stds, color=C_FL_ACC, alpha=0.15)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel(f"Mean {dist_label}  (modality heterogeneity)", labelpad=6)
    ax.set_ylabel("Score", labelpad=6)
    ax.tick_params(colors="#333", direction="out", length=4)
    ax2 = ax.twiny()
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xticks(dist_vals)
    ax2.set_xticklabels([str(a) for a in alphas], fontsize=8.5, color="#444")
    ax2.set_xlabel(f"Dirichlet α_modal\n[α_label = {cfg.ALPHA_LABEL_FIXED} fixed]",
                   fontsize=9, color="#444", labelpad=7)
    ax2.tick_params(colors="#444", direction="out", length=4)
    for sp in ax2.spines.values():
        sp.set_color("#AAAAAA")
    ax2.spines["top"].set_visible(True)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), borderaxespad=0,
                        frameon=True, framealpha=1.0, fancybox=False, edgecolor="#CCC",
                        fontsize=9, handlelength=2.0)
    legend.get_frame().set_linewidth(0.8)
    ax.set_title(f"Figure {fig_letter} — F1 & Accuracy vs {dist_label}  [RAVDESS]\n"
                 f"(FedArtML α={cfg.ALPHA_LABEL_FIXED} + FedAMM, mean ± std {len(cfg.SEEDS)} seeds)",
                 fontsize=11, fontweight="bold", pad=10)
    _savefig(fig, filename, cfg)
    plt.close(fig)


def plot_client_sweep_figure(results, level_key_prefix, levels, pal, dist_label_short,
                              fig_letter, filename, cl_f1, cl_acc, cfg):
    plt.rcParams.update(BASE_RC)
    fig, (ax_f1, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.8))
    plt.subplots_adjust(wspace=0.30, left=0.07, right=0.82, top=0.85, bottom=0.14)

    client_counts = results["client_counts"]
    x_pos = np.arange(len(client_counts))
    x_labels = [str(n) for n in client_counts]
    markers = ["o", "s", "^", "v"]

    for ax, metric, cl_val, ylabel, ptitle in [
        (ax_f1, "f1", cl_f1, "F1-Score", "(a) F1-Score"),
        (ax_acc, "acc", cl_acc, "Accuracy", "(b) Accuracy"),
    ]:
        ax.set_facecolor("white")
        cl_vals = np.array(results[f"CL_{metric}_mean"])
        ax.plot(x_pos, cl_vals, color=C_CL, lw=2.2, marker="D", markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.6, label="CL (compute-equalised)", zorder=3)
        for idx, level in enumerate(levels):
            key_mean = f"FL_{metric}_mean_{level_key_prefix}_{level:.2f}"
            key_std = f"FL_{metric}_std_{level_key_prefix}_{level:.2f}"
            if key_mean not in results:
                continue
            means = np.array(results[key_mean])
            stds = np.array(results[key_std])
            if len(means) != len(x_pos):
                print(f"  [warn] {key_mean} has {len(means)} entries but client_counts has "
                      f"{len(x_pos)} — skipping this line")
                continue
            c = pal[idx % len(pal)]
            ax.plot(x_pos, means, color=c, lw=1.8, marker=markers[idx % len(markers)], markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, linestyle="--",
                    label=f"FL  {dist_label_short}={level:.2f}", zorder=2)
            ax.fill_between(x_pos, means - stds, means + stds, color=c, alpha=0.12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0.0, cl_val + 0.12)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(direction="out", length=4)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_title(ptitle, fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Clients", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)

    handles, labels = ax_acc.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.83, 0.50),
                         borderaxespad=0, frameon=True, framealpha=1.0, fancybox=False,
                         edgecolor="#CCC", fontsize=9, handlelength=2.0)
    legend.get_frame().set_linewidth(0.8)
    fig.suptitle(f"Figure {fig_letter} — F1 & Accuracy vs Number of Clients  [RAVDESS]\n"
                 f"(fixed modality {dist_label_short}, FedArtML α_label={cfg.ALPHA_LABEL_FIXED}, "
                 f"mean ± std {len(cfg.SEEDS)} seeds)", fontsize=10, fontweight="bold", y=1.01)
    _savefig(fig, filename, cfg)
    plt.close(fig)


def save_results(cl_f1, cl_acc, sweep_results_full, results_jsd, results_hd, out_path, cfg):
    print("\n" + "═" * 60)
    print("STEP 6 — Saving results")
    print("═" * 60)
    lines = [
        "# RAVDESS results — FedArtML (label) + FedAMM (modality)",
        f"# alpha_label FIXED = {cfg.ALPHA_LABEL_FIXED}",
        f"# Seeds: {cfg.SEEDS}",
        "",
        f"CL_F1  = {round(cl_f1, 4)}",
        f"CL_ACC = {round(cl_acc, 4)}",
        "",
        f"ALPHA_LABEL_FIXED = {cfg.ALPHA_LABEL_FIXED}",
        "",
        f"SWEEP_RESULTS_FULL = {json.dumps(sweep_results_full, indent=4)}",
        "",
        f"RESULTS_VS_CLIENTS_JSD = {json.dumps(results_jsd, indent=4)}",
        "",
        f"RESULTS_VS_CLIENTS_HD  = {json.dumps(results_hd, indent=4)}",
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved to: {out_path}")


# ═════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RAVDESS multimodal FL pipeline")
    parser.add_argument("--ravdess-path", default="./RAVDESS",
                         help="Path to the downloaded RAVDESS dataset (run download_ravdess.py first)")
    parser.add_argument("--cache-path", default="./ravdess_features",
                         help="Where to cache extracted audio/image features")
    parser.add_argument("--images-dir", default="./images_ravdess",
                         help="Where to save output figures")
    parser.add_argument("--checkpoint-path", default="./checkpoints/ravdess_client_sweep_ckpt_local1.pkl",
                         help="Client-sweep checkpoint (auto-resumes if this file exists)")
    parser.add_argument("--results-out", default="./results_output_ravdess.py",
                         help="Where to write the final results summary")
    parser.add_argument("--device", default=None,
                         help="Which device to use, e.g. 'cuda:0', 'cuda:1', or 'cpu'. "
                              "Defaults to 'cuda' (first visible GPU) if available, else 'cpu'.")
    args = parser.parse_args()

    cfg = build_config(args)

    torch.manual_seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    print("=" * 60)
    print("RAVDESS — Multimodal FL Pipeline (fresh start)")
    print(f"  Device : {cfg.DEVICE}")
    if cfg.DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  Seeds  : {cfg.SEEDS}")
    print("=" * 60)

    print("\nLoading RAVDESS")
    (img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te) = load_ravdess(cfg.RAVDESS_PATH, cfg.CACHE_PATH, cfg)

    print("\nTraining RAVDESS (centralised baseline)")
    cl_f1, cl_acc = train_centralised(img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cfg)

    # Step 3 — standalone alpha sweep (independent of Step 4)
    print("\nRunning Step 3 (alpha sweep)")
    sweep_results_full = run_alpha_sweep_full(
        img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cl_f1, cl_acc, cfg, checkpoint_path="./checkpoints/ravdess_alpha_sweep_ckpt.pkl")

    plot_alpha_sweep_figure(sweep_results_full, "modal_jsd_mean", "Jensen-Shannon Distance",
                             "B", "figB_alpha_sweep_jsd.png", cl_f1, cl_acc, cfg)
    plot_alpha_sweep_figure(sweep_results_full, "modal_hd_mean", "Hellinger Distance",
                             "C", "figC_alpha_sweep_hd.png", cl_f1, cl_acc, cfg)

    print(f"\nfixed JSD levels = {cfg.FIXED_JSD_LEVELS}")
    print(f"fixed HD levels  = {cfg.FIXED_HD_LEVELS}")

    # Step 4 — client sweep + plots
    results_jsd_tail, results_hd_tail = run_client_sweep(
        img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cl_f1, cl_acc,
        fixed_jsd_levels=cfg.FIXED_JSD_LEVELS, fixed_hd_levels=cfg.FIXED_HD_LEVELS,
        cfg=cfg, checkpoint_path=args.checkpoint_path)

    plot_client_sweep_figure(results_jsd_tail, "JSD", cfg.FIXED_JSD_LEVELS, JSD_PAL, "JSD",
                              "D1", "figD1_client_sweep_jsd.png", cl_f1, cl_acc, cfg)
    plot_client_sweep_figure(results_hd_tail, "HD", cfg.FIXED_HD_LEVELS, HD_PAL, "HD",
                              "D2", "figD2_client_sweep_hd.png", cl_f1, cl_acc, cfg)

    print("\nCalibration check (JSD):")
    for j in cfg.FIXED_JSD_LEVELS:
        print(f"  target={j:.2f}  achieved_mean_per_client_count="
              f"{results_jsd_tail[f'achieved_jsd_mean_{j:.2f}']}")
    print("\nCalibration check (HD):")
    for h in cfg.FIXED_HD_LEVELS:
        print(f"  target={h:.2f}  achieved_mean_per_client_count="
              f"{results_hd_tail[f'achieved_hd_mean_{h:.2f}']}")

    save_results(cl_f1, cl_acc, sweep_results_full, results_jsd_tail, results_hd_tail,
                 args.results_out, cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()