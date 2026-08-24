"""
download_data.py — fetch the CREMA-D dataset (audio + video) onto the server.

Run this ONCE before train_cremad.py. Two independent download methods are
provided (same two options your notebook had) — use whichever works from
your friend's server.

Usage
-----
    # Option A: Zenodo (no account needed)
    python download_data.py --method zenodo --out ./CREMA-D

    # Option B: Kaggle (needs Kaggle API credentials, see below)
    python download_data.py --method kaggle --out ./CREMA-D

Kaggle credentials
-------------------
Do NOT hardcode your Kaggle key in this file or anywhere in the repo.
Instead, either:
  1. Place a kaggle.json file at ~/.kaggle/kaggle.json (standard Kaggle CLI
     way — download it from kaggle.com -> Account -> Create New API Token), or
  2. Export two environment variables before running this script:
        export KAGGLE_USERNAME=your_username
        export KAGGLE_KEY=your_key

NOTE: your original notebook set an env var called KAGGLE_API_TOKEN with a
hardcoded key. That variable name isn't actually read by the Kaggle CLI
(it expects KAGGLE_USERNAME / KAGGLE_KEY or the kaggle.json file), so that
line likely wasn't doing anything useful. Also, that key was a real,
exposed credential — please rotate it on kaggle.com since it's now been
shared in plaintext.
"""

import argparse
import os
import shutil
import subprocess
import sys


def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def download_zenodo(out_dir):
    tmp_zip_audio = "/tmp/AudioWAV.zip"
    tmp_zip_video = "/tmp/VideoFlash.zip"

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "zenodo-get"])
    run(["wget", "-q", "https://zenodo.org/record/1309958/files/AudioWAV.zip", "-O", tmp_zip_audio])
    run(["wget", "-q", "https://zenodo.org/record/1309958/files/VideoFlash.zip", "-O", tmp_zip_video])
    run(["unzip", "-q", tmp_zip_audio, "-d", out_dir])
    run(["unzip", "-q", tmp_zip_video, "-d", out_dir])

    n_audio = len([f for f in os.listdir(os.path.join(out_dir, "AudioWAV")) if f.endswith(".wav")])
    print(f"  Audio files: {n_audio}")
    if os.path.isdir(os.path.join(out_dir, "VideoFlash")):
        n_video = len([f for f in os.listdir(os.path.join(out_dir, "VideoFlash")) if f.endswith(".flv")])
        print(f"  Video files: {n_video}")


def download_kaggle(out_dir):
    have_env_creds = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    have_json = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    if not (have_env_creds or have_json):
        sys.exit(
            "No Kaggle credentials found. Set KAGGLE_USERNAME + KAGGLE_KEY "
            "env vars, or place ~/.kaggle/kaggle.json. See the docstring "
            "at the top of this file."
        )

    run([sys.executable, "-m", "pip", "install", "-q", "--break-system-packages", "kaggle"])

    raw_dir = "/tmp/cremad_raw"
    video_dir = "/tmp/cremad_video"
    run(["kaggle", "datasets", "download", "-d", "ejlok1/cremad", "-p", raw_dir])
    run(["kaggle", "datasets", "download", "-d", "stefanogiannini/crema-d-video", "-p", video_dir])

    os.makedirs(out_dir, exist_ok=True)
    run(["unzip", "-q", os.path.join(raw_dir, "cremad.zip"), "-d", out_dir])
    run(["unzip", "-q", os.path.join(video_dir, "crema-d-video.zip"), "-d", out_dir])

    # The Kaggle video dataset sometimes drops .flv files at the top level
    # instead of inside a VideoFlash/ subfolder — same fixup your notebook did.
    video_flash = os.path.join(out_dir, "VideoFlash")
    os.makedirs(video_flash, exist_ok=True)
    flv_files = [f for f in os.listdir(out_dir) if f.endswith(".flv")]
    if flv_files:
        print(f"  Moving {len(flv_files)} loose .flv files into VideoFlash/ ...")
        for f in flv_files:
            shutil.move(os.path.join(out_dir, f), os.path.join(video_flash, f))

    if os.path.isdir(os.path.join(out_dir, "AudioWAV")):
        print(f"  Audio files: {len(os.listdir(os.path.join(out_dir, 'AudioWAV')))}")
    print(f"  Video files: {len(os.listdir(video_flash))}")


def main():
    parser = argparse.ArgumentParser(description="Download CREMA-D dataset")
    parser.add_argument("--method", choices=["zenodo", "kaggle"], required=True)
    parser.add_argument("--out", default="./CREMA-D", help="Output directory")
    args = parser.parse_args()

    if args.method == "zenodo":
        download_zenodo(args.out)
    else:
        download_kaggle(args.out)

    print(f"\nDone. Dataset is at: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
