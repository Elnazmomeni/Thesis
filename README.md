# CREMA-D Multimodal FL Pipeline — server version

This is your `finalscript.ipynb` notebook converted into plain Python scripts
so it can run unattended on a remote GPU server (no Colab, no notebook UI
needed).

## Project layout

```
cremad_fl_project/
├── fedartml_local/          # your support modules (unchanged, just renamed
│   │                          off "__2_"/"__3_" suffixes)
│   ├── fl_modality_heterogeneity.py   ← used
│   ├── fedartml_patch.py              ← used
│   ├── fl_split_as_federated_data.py  ← kept for reference, not imported
│   ├── fl_interactive_plots.py        ← kept for reference, not imported
│   ├── function_base.py               ← kept for reference, not imported
│   └── __init__.py
├── download_data.py         # one-time dataset download
├── train_cremad.py          # the actual pipeline (config, features, model,
│                               FL training, sweeps, plots)
├── requirements.txt
└── README.md
```

Only `fl_modality_heterogeneity.py` and `fedartml_patch.py` are actually
imported by `train_cremad.py` — same as in your notebook (cell 2 only ever
imports those two). The other three files in `fedartml_local/` aren't used
by the pipeline itself; I kept them in case you need them for something
else, but you can delete them if you want a leaner setup.

## 1. Set up the environment on the server

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If `pip install` complains about externally-managed environments, add
`--break-system-packages`.

## 2. Download the dataset (once)

Pick whichever source actually works from that server:

```bash
# Option A — Zenodo, no account needed
python download_data.py --method zenodo --out ./CREMA-D

# Option B — Kaggle
mkdir -p ~/.kaggle
echo YOUR_KAGGLE_TOKEN > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
python download_data.py --method kaggle --out ./CREMA-D
```

Get `YOUR_KAGGLE_TOKEN` from kaggle.com → Settings → API → "Create New
Token" (it looks like `KGAT_...`). Generate it and paste it into the file
directly on the server — never hardcode it in a script or share it in a
chat/message, since anywhere it's been typed in plaintext should be
considered exposed and rotated.

**About the key that was in your original notebook**: it was hardcoded
directly in the code (`os.environ['KAGGLE_API_TOKEN'] = '...'`), which
means it's already been exposed. Please expire it at kaggle.com → Settings
→ API and generate a fresh one to use going forward. `download_data.py`
also accepts `export KAGGLE_API_TOKEN=your_token` as an environment
variable if you'd rather not use the file.

## 3. Run training

```bash
python train_cremad.py --cremad-path ./CREMA-D
```

Since this can run for hours, use `tmux` or `screen` so it survives you
disconnecting:

```bash
tmux new -s cremad
python train_cremad.py --cremad-path ./CREMA-D
# Ctrl+b, then d to detach
# later: tmux attach -t cremad
```

All CLI options (all optional, shown with their defaults):

| Flag | Default | What it does |
|---|---|---|
| `--cremad-path` | `./CREMA-D` | Dataset location |
| `--cache-path` | `./cremad_features` | Where extracted features are cached (`.npy`) so re-runs skip feature extraction |
| `--images-dir` | `./images_cremad_v6` | Where output figures are saved |
| `--checkpoint-path` | `./checkpoints/client_sweep_ckpt.pkl` | Client-sweep checkpoint — the script **auto-resumes** from here if interrupted |
| `--results-out` | `./results_output_cremad_v6.py` | Final results summary file |
| `--device` | auto (`cuda` if available, else `cpu`) | Which GPU to use, e.g. `cuda:0` or `cuda:1` on a multi-GPU box, or `cpu` |

## Picking a GPU on a multi-GPU server

If the server has more than one GPU, tell PyTorch which one to use with
`--device` instead of editing the script:

```bash
python train_cremad.py --cremad-path ./CREMA-D --device cuda:0
python train_cremad.py --cremad-path ./CREMA-D --device cuda:1
```

Run `nvidia-smi` on the server first to see which GPUs exist and how busy
each one is, so you don't accidentally pile two jobs onto the same card.

## What the script actually runs

Same sequence as your notebook cells 17–20:
1. Load/cache CREMA-D features (audio MFCCs + one video frame per clip)
2. Train the centralised baseline model
3. Run the checkpointed client sweep (varies client count × fixed JSD/HD
   heterogeneity levels, 5 seeds each) via FedAvg
4. Save the two sweep figures + a results summary `.py` file

## Things carried over as-is from your notebook (flagged, not silently fixed)

- **`run_alpha_sweep_small()`** calls a function `downsample_per_class(...)`
  that isn't defined anywhere in your notebook. It's dead code — your
  actual run (cells 17–20) never calls this function, only
  `run_client_sweep()`, so it never executes. If you do want to call it,
  you'll need to write `downsample_per_class` first.
- **`plot_client_sweep_figure` is defined twice** in your notebook (once in
  cell 15, once in cell 16), and the second definition silently replaces
  the first at import time — same in `train_cremad.py`. The cell-16 version
  that ends up active reads the x-axis from the global `CLIENT_SWEEP`
  list rather than from `results["client_counts"]` like the cell-15
  version does (which the cell-15 comment calls a deliberate "FIX"). I
  preserved this exact behavior rather than picking one for you, since I
  don't know which one you intended to keep.

## Small environment fixes I did make

- Added `matplotlib.use("Agg")` before importing `pyplot` — required on a
  headless server with no display, otherwise matplotlib can crash on import.
- Switched `opencv-python` → `opencv-python-headless` in requirements — no
  GUI dependencies, standard practice for a server.
- Removed the `google.colab` mount and the in-notebook `!pip install` /
  `!wget` shell commands — replaced by `requirements.txt` and
  `download_data.py`.
