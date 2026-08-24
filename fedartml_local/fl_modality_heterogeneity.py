"""
fl_modality_heterogeneity.py
────────────────────────────
Generic modality-heterogeneity class for FedArtML.
Works with ANY multimodal dataset whose modalities are separate numpy arrays
(e.g. AVMNIST image + audio, BraTS MRI sequences, etc.).

No PyTorch tensors. No dataset-specific transforms.

PLACE THIS FILE at:
    fedartml/fl_modality_heterogeneity.py

Then add ONE line to fedartml/__init__.py:
    from fedartml.fl_modality_heterogeneity import ModalityHeterogeneity

CONCEPT
-------
Each sample can have some modalities *present* (kept) or *absent* (zeroed out).
Presence/absence patterns are drawn from a Dirichlet distribution per client,
mirroring FedAMM's mask_id_count / client_mask_id_proportions scheme.

    alpha → large (e.g. 100) : all clients see similar patterns  →  IID
    alpha → small (e.g. 0.1) : clients specialise in few patterns → non-IID

For AVMNIST (2 modalities: image + audio) there are 2² - 1 = 3 non-empty
patterns: image-only, audio-only, both.
"""

import numpy as np
import pandas as pd
from itertools import combinations


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_all_masks(num_modalities: int):
    """Return all 2^n - 1 non-empty boolean mask tuples for n modalities."""
    masks = []
    for r in range(1, num_modalities + 1):
        for combo in combinations(range(num_modalities), r):
            mask = [False] * num_modalities
            for idx in combo:
                mask[idx] = True
            masks.append(tuple(mask))
    return masks


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    m = 0.5 * (p + q)

    def _kl(a, b):
        ix = a > 1e-10
        return float(np.sum(a[ix] * np.log(a[ix] / np.maximum(b[ix], 1e-10))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance between two probability vectors."""
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))



# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ModalityHeterogeneity:
    """
    Simulate modality heterogeneity across federated clients for any
    multimodal dataset whose modalities are separate numpy arrays.

    Parameters
    ----------
    modality_names : list of str
        Ordered names of the modalities, e.g. ['image', 'audio'] for AVMNIST
        or ['T1', 'T1ce', 'T2', 'FLAIR'] for BraTS.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    num_modalities : int
        Number of modalities inferred from modality_names.
    all_masks : list of tuple of bool
        All 2^num_modalities - 1 non-empty presence/absence patterns.
    mask_names : list of str
        Human-readable name for each mask pattern, e.g. 'image+audio'.

    Examples
    --------
    >>> from fedartml import ModalityHeterogeneity
    >>> mh = ModalityHeterogeneity(['image', 'audio'], random_state=42)
    >>> modality_arrays = {'image': X_image_train, 'audio': X_audio_train}
    >>> client_data, report = mh.assign_modalities_to_clients(
    ...     modality_arrays, y_train, num_clients=4, alpha=0.5
    ... )
    >>> mh.summary(client_data)
    """

    def __init__(self, modality_names: list, random_state=None):
        if len(modality_names) < 1:
            raise ValueError("modality_names must have at least one entry.")
        self.modality_names  = list(modality_names)
        self.num_modalities  = len(modality_names)
        self.random_state    = random_state
        self.all_masks       = _build_all_masks(self.num_modalities)
        self.mask_names      = self._build_mask_names()

    # ── private helpers ────────────────────────────────────────────────────────

    def _build_mask_names(self) -> list:
        return [
            "+".join(self.modality_names[i] for i, v in enumerate(mask) if v)
            for mask in self.all_masks
        ]

    # ── public API ─────────────────────────────────────────────────────────────

    def assign_modalities_to_clients(
        self,
        modality_arrays: dict,
        y: np.ndarray,
        num_clients: int = 4,
        alpha: float = 1.0,
        prefix_cli: str = "client",
    ) -> tuple:
        """
        Assign per-sample modality availability masks to each client using
        a Dirichlet distribution over all non-empty mask patterns.

        Each client gets ALL N samples but with absent modalities zeroed out,
        matching FedAMM's approach of mask-based heterogeneity simulation.

        Parameters
        ----------
        modality_arrays : dict {str: np.ndarray}
            One key per modality (must match self.modality_names).
            Each value is a numpy array of shape (N, ...) — any number of
            trailing dimensions is supported (images, spectrograms, flat, etc.).
        y : np.ndarray, shape (N,)
            Class labels.
        num_clients : int
            Number of federated clients.
        alpha : float
            Dirichlet concentration parameter.
            Large (e.g. 100)  → similar distributions across clients  (IID).
            Small (e.g. 0.01) → each client specialises in few patterns (non-IID).
        prefix_cli : str
            Prefix for client keys, e.g. 'client' → 'client_1', 'client_2', ...

        Returns
        -------
        client_data : dict
            Keys = client names (e.g. 'client_1').
            Each value is a dict:
                '<modality_name>' : np.ndarray (N, ...)
                    The modality array with absent samples zeroed out.
                'y'               : np.ndarray (N,)  — labels
                'masks'           : bool np.ndarray (N, num_modalities)
                                    True = modality present for that sample.
                'mask_ids'        : int np.ndarray (N,)
                                    Index into self.all_masks per sample.
        mask_report : pd.DataFrame
            Per-client summary: modality availability rates and pattern counts.

        Raises
        ------
        ValueError
            If modality_arrays keys do not match self.modality_names, or if
            arrays have inconsistent first-dimension size.
        """
        # ── Validate inputs ──────────────────────────────────────────────────
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

        # Dirichlet proportions: shape (num_clients, num_masks)
        dirichlet_props = rng.dirichlet(
            alpha=np.full(num_masks, alpha), size=num_clients
        )

        client_data = {}
        report_rows = []

        for c_idx, c_name in enumerate(client_names):

            # ── Draw a mask pattern index for every sample ───────────────────
            mask_ids   = rng.choice(num_masks, size=N, p=dirichlet_props[c_idx])

            # Boolean mask: (N, num_modalities) — True = modality present
            masks_bool = np.array(
                [self.all_masks[mid] for mid in mask_ids], dtype=bool
            )  # shape (N, M)

            # ── Zero-out absent modalities — works for any array shape ───────
            client_entry = {"y": y.copy(), "masks": masks_bool, "mask_ids": mask_ids}

            for m_idx, mod_name in enumerate(self.modality_names):
                arr     = modality_arrays[mod_name].copy().astype(np.float32)
                present = masks_bool[:, m_idx]           # (N,) bool

                # Build a broadcast-safe shape: (N, 1, 1, ...) for the mask
                extra_dims   = arr.ndim - 1              # number of trailing dims
                mask_shape   = (N,) + (1,) * extra_dims
                arr         *= present.reshape(mask_shape).astype(np.float32)

                client_entry[mod_name] = arr

            client_data[c_name] = client_entry

            # ── Build report row ─────────────────────────────────────────────
            unique_ids, counts = np.unique(mask_ids, return_counts=True)
            mod_avail = masks_bool.mean(axis=0)          # (M,)

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

    # ── heterogeneity metrics ──────────────────────────────────────────────────

    def compute_modality_heterogeneity_score(self, client_data: dict) -> dict:
        """
        Compute pairwise Jensen-Shannon divergence and Hellinger distance
        between clients' mask-pattern distributions.

        Parameters
        ----------
        client_data : dict
            Output of assign_modalities_to_clients().

        Returns
        -------
        dict with:
            'js_divergence_matrix'   : pd.DataFrame (C x C)
            'hellinger_matrix'       : pd.DataFrame (C x C)
            'mean_js_divergence'     : float  (0 = IID, higher = more non-IID)
            'mean_hellinger'         : float  (0 = IID, higher = more non-IID)
            'modality_availability'  : dict  client → {modality: availability rate}
        """
        num_masks    = len(self.all_masks)
        client_names = list(client_data.keys())
        C            = len(client_names)

        # Per-client probability vector over mask patterns
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

    def get_client_modal_weights(self, client_data: dict,
                                  client_order: list = None) -> np.ndarray:
        """
        Column-normalised per-client modality availability weights.

        Mirrors client_modal_weight in FedAMM train.py.
        Use these for weighted FedAvg aggregation of per-modality encoders.

        Parameters
        ----------
        client_data : dict
            Output of assign_modalities_to_clients().
        client_order : list of str, optional
            Explicit client name order to use for the returned rows.
            CRITICAL: this must match the exact order used to build the
            corresponding client_datasets list in the training loop, or
            row i of the returned array will be paired with the wrong
            client during FedAvg aggregation. If not provided, defaults
            to client_data.keys() insertion order (NOT sorted() — using
            sorted() on names like "client_10" vs "client_2" silently
            reorders clients lexicographically once you have 10+
            clients, which will not match insertion order and corrupts
            the weighting if the caller built client_datasets a
            different way).

        Returns
        -------
        ndarray (num_clients, num_modalities)
            Each column sums to 1 across clients, in `client_order`.
        """
        client_names = client_order if client_order is not None \
            else list(client_data.keys())
        raw = np.array(
            [client_data[c]["masks"].sum(axis=0) for c in client_names],
            dtype=np.float32,
        )
        col_sums = raw.sum(axis=0)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        return raw / col_sums

    # ── diagnostics ───────────────────────────────────────────────────────────

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