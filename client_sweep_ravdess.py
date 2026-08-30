"""
run_step4_only.py — resume the RAVDESS pipeline starting at Step 4
(client sweep), skipping Step 2 (CL training) and Step 3 (alpha sweep)
since those already completed in a previous run.

Usage (same flags as train_ravdess.py, minus the ones we don't need):
    python run_step4_only.py --ravdess-path ./RAVDESS --images-dir ./images_ravdess

Step 4 (run_client_sweep) already checkpoints itself to --checkpoint-path
and resumes automatically, so if it crashed partway through, just re-run
this script and it will pick up from the last completed
(n_clients, jsd/hd, level) combo.

IMPORTANT: edit CL_F1 / CL_ACC below if your Step 2 numbers were
different from the ones you already have.
"""

import argparse
import torch
import numpy as np

# Reuse everything from the original script — must be in the same directory.
# NOTE: change this import if your RAVDESS training module has a different
# filename (this mirrors train_cremad.py from the CREMA-D pipeline).
import train_ravdess as tr

# ─── Hardcode the Step 2 result you already have, instead of retraining ───
CL_F1 = 0.9064
CL_ACC = 0.9070


def main():
    parser = argparse.ArgumentParser(description="Resume RAVDESS pipeline at Step 4 only")
    parser.add_argument("--ravdess-path", default="./RAVDESS")
    parser.add_argument("--cache-path", default="./ravdess_features")
    parser.add_argument("--images-dir", default="./images_ravdess")
    parser.add_argument("--checkpoint-path", default="./checkpoints/ravdess_client_sweep_ckpt.pkl")
    parser.add_argument("--results-out", default="./results_output_ravdess.py")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = tr.build_config(args)

    torch.manual_seed(cfg.RANDOM_STATE)
    np.random.seed(cfg.RANDOM_STATE)

    print("=" * 60)
    print("RAVDESS — resuming at STEP 4 only (client sweep)")
    print(f"  Device : {cfg.DEVICE}")
    print(f"  Using cached CL result: F1={CL_F1}  Acc={CL_ACC}")
    print("=" * 60)

    # Step 1 — load data (this hits your feature cache on disk, so it's fast,
    # NOT a re-extraction — as long as ravdess_features/img.npy etc. exist)
    print("\nLoading RAVDESS (from feature cache)")
    (img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te) = tr.load_ravdess(
        cfg.RAVDESS_PATH, cfg.CACHE_PATH, cfg)

    cl_f1, cl_acc = CL_F1, CL_ACC

    # Step 3's fixed JSD/HD levels — same values as in the original main()
    print(f"fixed JSD levels = {cfg.FIXED_JSD_LEVELS}")
    print(f"fixed HD levels  = {cfg.FIXED_HD_LEVELS}")

    # Step 4 — client sweep (auto-resumes from checkpoint if it exists)
    results_jsd_tail, results_hd_tail = tr.run_client_sweep(
        img_tr, aud_tr, lbl_tr, img_te, aud_te, lbl_te, cl_f1, cl_acc,
        fixed_jsd_levels=cfg.FIXED_JSD_LEVELS, fixed_hd_levels=cfg.FIXED_HD_LEVELS,
        cfg=cfg, checkpoint_path=args.checkpoint_path)

    tr.plot_client_sweep_figure(results_jsd_tail, "JSD", cfg.FIXED_JSD_LEVELS, tr.JSD_PAL, "JSD",
                                 "D1", "figD1_client_sweep_jsd.png", cl_f1, cl_acc, cfg)
    tr.plot_client_sweep_figure(results_hd_tail, "HD", cfg.FIXED_HD_LEVELS, tr.HD_PAL, "HD",
                                 "D2", "figD2_client_sweep_hd.png", cl_f1, cl_acc, cfg)

    print("\nCalibration check (JSD):")
    for j in cfg.FIXED_JSD_LEVELS:
        print(f"  target={j:.2f}  achieved_mean_per_client_count="
              f"{results_jsd_tail[f'achieved_jsd_mean_{j:.2f}']}")
    print("\nCalibration check (HD):")
    for h in cfg.FIXED_HD_LEVELS:
        print(f"  target={h:.2f}  achieved_mean_per_client_count="
              f"{results_hd_tail[f'achieved_hd_mean_{h:.2f}']}")

    # save_results in the original pipeline expects a sweep_results_full
    # argument (from Step 3). Since Step 3 is skipped here, pass an empty
    # list — matches how the RAVDESS main() handled the --skip-step3 case.
    tr.save_results(cl_f1, cl_acc, [], results_jsd_tail, results_hd_tail,
                     args.results_out, cfg)

    print("\nAll done.")


if __name__ == "__main__":
    main()