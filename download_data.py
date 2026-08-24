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
Do NOT hardcode your Kaggle token in this file or anywhere in the repo.
Kaggle's current API uses a single access token (looks like "KGAT_..."),
generated from kaggle.com -> Settings -> API -> "Create New Token". Use it
one of two ways:
  1. Save it to a file at ~/.kaggle/access_token:
        mkdir -p ~/.kaggle
        echo YOUR_TOKEN > ~/.kaggle/access_token
        chmod 600 ~/.kaggle/access_token
  2. Or export it as an environment variable before running this script:
        export KAGGLE_API_TOKEN=YOUR_TOKEN

Either way, generate the token directly on the server (or paste it into a
file/terminal there) — never hardcode it in a script or paste it into a
chat, since anywhere it's been typed in plaintext should be treated as
exposed and rotated.

(Older Kaggle accounts may instead use a kaggle.json with separate
"username"/"key" fields under ~/.kaggle/kaggle.json — that also still works
and doesn't need any extra setup here.)
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
    have_token_env = "KAGGLE_API_TOKEN" in os.environ
    have_token_file = os.path.exists(os.path.expanduser("~/.kaggle/access_token"))
    have_json = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
    have_legacy_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    if not (have_token_env or have_token_file or have_json or have_legacy_env):
        sys.exit(
            "No Kaggle credentials found. Set KAGGLE_API_TOKEN, or place "
            "~/.kaggle/access_token (new token-based auth), or "
            "~/.kaggle/kaggle.json (legacy username/key auth). "
            "See the docstring at the top of this file."
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
