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
