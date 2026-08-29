"""
download_ravdess.py — downloads and extracts the RAVDESS Video_Speech
dataset (24 actors) from Zenodo.

Usage:
    python download_ravdess.py --output-path ./RAVDESS

Safe to re-run: already-extracted actors are skipped, and failed
downloads are retried with exponential backoff.
"""

import argparse
import os
import time
import urllib.request
import zipfile


def download_actor(actor_num, output_path, retries=3, delay=5):
    actor_str = f"{actor_num:02d}"
    zip_name = f"Video_Speech_Actor_{actor_str}.zip"
    zip_path = f"/tmp/{zip_name}"
    actor_dir = os.path.join(output_path, f"Actor_{actor_str}")

    if os.path.isdir(actor_dir) and len(os.listdir(actor_dir)) > 0:
        print(f"  Actor {actor_str}: already done, skipping")
        return True

    url = f"https://zenodo.org/record/1188976/files/{zip_name}"

    for attempt in range(1, retries + 1):
        try:
            print(f"  Actor {actor_str}/24 — attempt {attempt}...", end=" ", flush=True)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(output_path)
            os.remove(zip_path)
            print("done")
            time.sleep(1)  # be polite to Zenodo
            return True
        except Exception as e:
            print(f"FAILED: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if attempt < retries:
                print(f"    Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2  # exponential backoff
    return False


def main():
    parser = argparse.ArgumentParser(description="Download the RAVDESS Video_Speech dataset")
    parser.add_argument("--output-path", default="./RAVDESS",
                         help="Where to extract the dataset (default: ./RAVDESS)")
    parser.add_argument("--num-actors", type=int, default=24,
                         help="Number of actors to download (default: 24, the full dataset)")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    failed = []
    for i in range(1, args.num_actors + 1):
        ok = download_actor(i, args.output_path)
        if not ok:
            failed.append(i)

    done = len([d for d in os.listdir(args.output_path) if d.startswith("Actor_")])
    print(f"\nDone: {done}/{args.num_actors} actors extracted to {args.output_path}")
    if failed:
        print(f"Failed actors (re-run this script to retry): {failed}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()