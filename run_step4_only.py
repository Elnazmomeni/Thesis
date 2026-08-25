"""
run_step4_only.py — resume the CREMA-D pipeline starting at Step 4
(client sweep), skipping Step 2 (CL training) and Step 3 (alpha sweep)
since those already completed in a previous run.

Usage (same flags as train_cremad.py, minus the ones we don't need):
    python run_step4_only.py --cremad-path ./CREMA-D --images-dir ./images_cremad_v6

Step 4 (run_client_sweep) already checkpoints itself to --checkpoint-path
and resumes automatically, so if it crashed partway through, just re-run
this script and it will pick up from the last completed
(n_clients, jsd/hd, level) combo.

IMPORTANT: edit CL_F1 / CL_ACC below if your Step 2 numbers were
different from the ones in your log ("CL final:  F1=0.5859   Acc=0.5857").
"""

import argparse
import torch
import numpy as np

# Reuse everything from the original script — must be in the same directory.
import train_cremad as tc

# ─── Hardcode the Step 2 result you already have, instead of retraining ───
CL_F1 = 0.5859
CL_ACC = 0.5857


def main():
    parser = argparse.ArgumentParser(description="Resume CREMA-D pipeline at Step 4 only")
    parser.add_argument("--cremad-path", default="./CREMA-D")
    parser.add_argument("--cache-path", default="./cremad_features")
    parser.add_argument("--images-dir", default="./images_cremad_v6")
    parser.add_argument("--checkpoint-path", default="./checkpoints/client_sweep_ckpt.pkl")
    parser.add_argument("--results-out", default="./results_output_cremad_v6.py")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = tc.build_config(args)

    torch.manual_seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    print("=" * 60)
    print("CREMA-D — resuming at STEP 4 only (client sweep)")
    print(f"  Device : {cfg.DEVICE}")
    print(f"  Using cached CL result: F1={CL_F1}  Acc={CL_ACC}")
    print("=" * 60)

    # Step 1 — load data (this hits your feature cache on disk, so it's fast,
    # NOT a re-extraction — as long as cremad_features/img.npy etc. exist)
    print("\nLoading CREMA-D (from feature cache)")
    (img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te) = tc.load_cremad(
        cfg.CREMAD_PATH, cfg.CACHE_PATH, cfg)

    cl_f1, cl_acc = CL_F1, CL_ACC

    # Step 3's fixed JSD/HD levels — same values as in the original main()
    cfg.FIXED_JSD_LEVELS = [0.02, 0.10, 0.24, 0.39, 0.48]
    cfg.FIXED_HD_LEVELS = [0.05, 0.16, 0.37, 0.57, 0.70]
    print(f"fixed JSD levels = {cfg.FIXED_JSD_LEVELS}")
    print(f"fixed HD levels  = {cfg.FIXED_HD_LEVELS}")

    # Step 4 — client sweep (auto-resumes from checkpoint if it exists)
    results_jsd_tail, results_hd_tail = tc.run_client_sweep(
        img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cl_f1, cl_acc,
        fixed_jsd_levels=cfg.FIXED_JSD_LEVELS, fixed_hd_levels=cfg.FIXED_HD_LEVELS,
        cfg=cfg, checkpoint_path=args.checkpoint_path)

    tc.plot_client_sweep_figure(results_jsd_tail, "JSD", cfg.FIXED_JSD_LEVELS, tc.JSD_PAL, "JSD",
                                 "D1-tail", "figD1_client_sweep_jsd_20_100.png", cl_f1, cl_acc, cfg)
    tc.plot_client_sweep_figure(results_hd_tail, "HD", cfg.FIXED_HD_LEVELS, tc.HD_PAL, "HD",
                                 "D2-tail", "figD2_client_sweep_hd_20_100.png", cl_f1, cl_acc, cfg)

    print("\nCalibration check (JSD):")
    for j in cfg.FIXED_JSD_LEVELS:
        print(f"  target={j:.2f}  achieved_mean_per_client_count="
              f"{results_jsd_tail[f'achieved_jsd_mean_{j:.2f}']}")
    print("\nCalibration check (HD):")
    for h in cfg.FIXED_HD_LEVELS:
        print(f"  target={h:.2f}  achieved_mean_per_client_count="
              f"{results_hd_tail[f'achieved_hd_mean_{h:.2f}']}")

    tc.save_results(cl_f1, cl_acc, results_jsd_tail, results_hd_tail, args.results_out, cfg=cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()