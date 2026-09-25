import pickle
import numpy as np

OLD_PATH = "ravdess_alpha_sweep_allseeds_moreseeds.pkl"      # your original 5-seed checkpoint
NEW_PATH = "ravdess_alpha_sweep_newseeds.pkl"       # the 5 new seeds you just ran
OUT_PATH = "ravdess_alpha_sweep_merged10.pkl"       # combined 10-seed result

with open(OLD_PATH, "rb") as f:
    old = pickle.load(f)["results_by_alpha"]
with open(NEW_PATH, "rb") as f:
    new = pickle.load(f)["results_by_alpha"]

# sanity check before merging anything
old_alphas = set(old.keys())
new_alphas = set(new.keys())
if old_alphas != new_alphas:
    missing_in_new = old_alphas - new_alphas
    missing_in_old = new_alphas - old_alphas
    raise ValueError(
        f"Alpha sets don't match between checkpoints — can't merge safely.\n"
        f"  In old but not new: {missing_in_new}\n"
        f"  In new but not old: {missing_in_old}"
    )

merged = {}
for a in sorted(old.keys()):
    o, n = old[a], new[a]
    assert o["alpha_label"] == n["alpha_label"], f"alpha_label mismatch at alpha={a}"

    m = {
        "alpha_modal": a,
        "alpha_label": o["alpha_label"],
        "seeds": o["seeds"] + n["seeds"],
    }
    for key in ["f1", "acc", "modal_jsd", "modal_hd", "label_jsd", "label_hd"]:
        m[key] = o[key] + n[key]
        m[f"{key}_mean"] = float(np.mean(m[key]))
        m[f"{key}_std"] = float(np.std(m[key]))
    merged[a] = m

with open(OUT_PATH, "wb") as f:
    pickle.dump({"results_by_alpha": merged, "done": set(merged.keys())}, f)

print(f"Merged checkpoint saved to {OUT_PATH}\n")
print(f"{'alpha_modal':>12} | {'n_seeds':>7} | {'f1_mean':>8} | {'f1_std':>7} | seeds")
print("-" * 90)
for a in sorted(merged.keys()):
    m = merged[a]
    print(f"{a:>12} | {len(m['seeds']):>7} | {m['f1_mean']:>8.4f} | {m['f1_std']:>7.4f} | {m['f1']}")