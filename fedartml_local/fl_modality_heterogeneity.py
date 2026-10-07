# imports

import numpy as np
import pandas as pd
from itertools import combinations


# helpers

# all 2^n - 1 non-empty boolean masks for n modalities
def _build_all_masks(num_modalities: int):
    masks = []
    for r in range(1, num_modalities + 1):
        for combo in combinations(range(num_modalities), r):
            mask = [False] * num_modalities
            for idx in combo:
                mask[idx] = True
            masks.append(tuple(mask))
    return masks

# jsd between two probability vectors
def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    
    m = 0.5 * (p + q)

    def _kl(a, b):
        ix = a > 1e-10
        return float(np.sum(a[ix] * np.log(a[ix] / np.maximum(b[ix], 1e-10))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

# hd between two probability vectors.
def _hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:

    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))



#  main class
 
# simulates modality heterogeneity across federated clients
class ModalityHeterogeneity:
 
    def __init__(self, modality_names: list, random_state=None):
        if len(modality_names) < 1:
            raise ValueError("modality_names must have at least one entry.")
        self.modality_names  = list(modality_names)
        self.num_modalities  = len(modality_names)
        self.random_state    = random_state
        self.all_masks       = _build_all_masks(self.num_modalities)
        self.mask_names      = self._build_mask_names()

    # private helpers

    def _build_mask_names(self) -> list:
        return [
            "+".join(self.modality_names[i] for i, v in enumerate(mask) if v)
            for mask in self.all_masks
        ]

    # public API

    def assign_modalities_to_clients(
        self,
        modality_arrays: dict,
        y: np.ndarray,
        num_clients: int = 4,
        alpha: float = 1.0,
        prefix_cli: str = "client",
    ) -> tuple:
        
        # validate inputs 
        missing = set(self.modality_names) - set(modality_arrays.keys())
        if missing:
            raise ValueError(
                f"modality_arrays is missing keys: {missing}. "
                f"Expected: {self.modality_names}"
            )

        N = len(y)
        for name in self.modality_names:
            arr = modality_arrays[name]
            if len(arr) != N:
                raise ValueError(
                    f"modality_arrays['{name}'] has {len(arr)} samples "
                    f"but y has {N}."
                )

        num_masks    = len(self.all_masks)
        rng          = np.random.default_rng(self.random_state)
        client_names = [f"{prefix_cli}_{i + 1}" for i in range(num_clients)]

        # dirichlet proportions: shape (num_clients, num_masks)
        dirichlet_props = rng.dirichlet(
            alpha=np.full(num_masks, alpha), size=num_clients
        )

        client_data = {}
        report_rows = []

        for c_idx, c_name in enumerate(client_names):

            # pick a mask pattern for every sample
            mask_ids   = rng.choice(num_masks, size=N, p=dirichlet_props[c_idx])

            # (N, num_modalities), True = modality present
            masks_bool = np.array(
                [self.all_masks[mid] for mid in mask_ids], dtype=bool
            ) 

            # zero out the absent modalities
            client_entry = {"y": y.copy(), "masks": masks_bool, "mask_ids": mask_ids}

            for m_idx, mod_name in enumerate(self.modality_names):
                arr     = modality_arrays[mod_name].copy().astype(np.float32)
                present = masks_bool[:, m_idx]          

                # reshape to (N, 1, 1, ...) so it broadcasts over any array shape
                extra_dims   = arr.ndim - 1              # number of trailing dims
                mask_shape   = (N,) + (1,) * extra_dims
                arr         *= present.reshape(mask_shape).astype(np.float32)

                client_entry[mod_name] = arr

            client_data[c_name] = client_entry

            # report row
            unique_ids, counts = np.unique(mask_ids, return_counts=True)
            mod_avail = masks_bool.mean(axis=0)     

            row = {
                "client":          c_name,
                "n_samples":       N,
                "unique_patterns": len(unique_ids),
            }
            for m, mod_name in enumerate(self.modality_names):
                row[f"{mod_name}_avail_rate"] = round(float(mod_avail[m]), 4)
            for mid, cnt in zip(unique_ids, counts):
                row[f"pattern_{self.mask_names[mid]}"] = int(cnt)

            report_rows.append(row)

        return client_data, pd.DataFrame(report_rows)

    # heterogeneity metrics 

    # compute pairwise jsd and hd between clients mask pattern distributions
    def compute_modality_heterogeneity_score(self, client_data: dict) -> dict:

        num_masks    = len(self.all_masks)
        client_names = list(client_data.keys())
        C            = len(client_names)

        # probability vector over mask patterns for each client
        dist_matrix = np.zeros((C, num_masks))
        for c_idx, c_name in enumerate(client_names):
            mask_ids = client_data[c_name]["mask_ids"]
            for mid in range(num_masks):
                dist_matrix[c_idx, mid] = np.sum(mask_ids == mid)
            dist_matrix[c_idx] /= dist_matrix[c_idx].sum()

        jsd_matrix = np.zeros((C, C))
        hd_matrix  = np.zeros((C, C))
        for i in range(C):
            for j in range(i + 1, C):
                jsd = _js_divergence(dist_matrix[i], dist_matrix[j])
                hd  = _hellinger_distance(dist_matrix[i], dist_matrix[j])
                jsd_matrix[i, j] = jsd_matrix[j, i] = jsd
                hd_matrix[i, j]  = hd_matrix[j, i]  = hd

        upper_idx = np.triu_indices(C, k=1)

        mod_avail = {
            c_name: {
                mod_name: float(client_data[c_name]["masks"][:, m].mean())
                for m, mod_name in enumerate(self.modality_names)
            }
            for c_name in client_names
        }

        return {
            "js_divergence_matrix":  pd.DataFrame(jsd_matrix,
                                                   index=client_names,
                                                   columns=client_names),
            "hellinger_matrix":      pd.DataFrame(hd_matrix,
                                                   index=client_names,
                                                   columns=client_names),
            "mean_js_divergence":    float(jsd_matrix[upper_idx].mean()),
            "mean_hellinger":        float(hd_matrix[upper_idx].mean()),
            "modality_availability": mod_avail,
        }

    # column-normalised modality availability weights per client
    def get_client_modal_weights(self, client_data: dict,
                                  client_order: list = None) -> np.ndarray:

        client_names = client_order if client_order is not None \
            else list(client_data.keys())
        raw = np.array(
            [client_data[c]["masks"].sum(axis=0) for c in client_names],
            dtype=np.float32,
        )
        col_sums = raw.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        return raw / col_sums

    # diagnostics 

     # prints a summary of the modality heterogeneity
    def summary(self, client_data: dict) -> None:
        """Print a human-readable modality heterogeneity summary."""
        scores  = self.compute_modality_heterogeneity_score(client_data)
        modal_w = self.get_client_modal_weights(client_data)
        names   = self.modality_names

        print("=" * 66)
        print("  ModalityHeterogeneity Summary")
        print("=" * 66)
        print(f"  Modalities         : {names}")
        print(f"  Clients            : {list(client_data.keys())}")
        print(f"  Non-empty patterns : {len(self.all_masks)}  "
              f"({self.mask_names})")
        print(f"  Mean JS Divergence : {scores['mean_js_divergence']:.4f}"
              f"  (0 = IID)")
        print(f"  Mean Hellinger     : {scores['mean_hellinger']:.4f}"
              f"  (0 = IID)")
        print()
        print("  Per-client modality availability rates:")
        for c_name, avail in scores["modality_availability"].items():
            rates = "   ".join(f"{k}: {v:.1%}" for k, v in avail.items())
            print(f"    {c_name:12s} →  {rates}")
        print()
        print("  Per-client modal weights (for weighted FedAvg):")
        for c_idx, c_name in enumerate(client_data.keys()):
            w    = modal_w[c_idx]
            wstr = "   ".join(
                f"{names[m]}: {w[m]:.3f}" for m in range(self.num_modalities)
            )
            print(f"    {c_name:12s} →  {wstr}")
        print("=" * 66)