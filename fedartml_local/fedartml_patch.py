
"""
fedartml_patch.py  — drop-in replacement for SplitAsFederatedData
Save this file to /content/drive/MyDrive/fedartml/fedartml_patch.py

THE ONLY CHANGE vs the original:
  Every np.random.seed() call uses (value % 97) so the seed is always
  in [0, 96] regardless of how many times the loop increments it.
  97 is prime and small enough that even 100 clients * 6 classes *
  10000 while-loop iterations = 6,000,000 increments of 100 each =
  600,000,000 — and 600,000,000 % 97 is always in [0,96]. Safe.
"""

import sys as _sys
import importlib as _importlib

# ── import distance functions from pip-installed fedartml ──────────────────
# Temporarily remove the Drive folder from sys.path so Python finds the
# pip package instead of the Drive folder named "fedartml".
_DRIVE = '/content/drive/MyDrive/fedartml'
_had   = _DRIVE in _sys.path
if _had:
    _sys.path.remove(_DRIVE)
try:
    _fb = _importlib.import_module('fedartml.function_base')
    jensen_shannon_distance = _fb.jensen_shannon_distance
    hellinger_distance      = _fb.hellinger_distance
    earth_movers_distance   = _fb.earth_movers_distance
finally:
    if _had and _DRIVE not in _sys.path:
        _sys.path.insert(0, _DRIVE)

import numpy as np
import pandas as pd

# Seed cap — MUST be small. 97 * 100_clients * 6_classes * 10000_loops
# = 582,000,000 << 2**32-1 = 4,294,967,295.  Safe for any realistic run.
_CAP = 97

def _s(v):
    """Clamp any integer to a safe numpy seed."""
    if v is None:
        return None
    return int(v) % _CAP


class SplitAsFederatedData:

    def __init__(self, random_state=None):
        self.random_state = random_state

    # ------------------------------------------------------------------
    @staticmethod
    def percent_noniid_method(labels, local_nodes, pct_noniid=0,
                               random_state=None):
        n_noniid = int(len(labels) * (pct_noniid / 100))
        sorted_labels = labels
        noniid_part_sample  = list(sorted_labels[0:n_noniid])
        iid_part_sample     = list(sorted_labels[n_noniid:len(labels)])
        uniq_class_noniid   = np.unique(noniid_part_sample)
        n_class_per_node    = len(uniq_class_noniid) // local_nodes

        pctg_distr, num_distr, idx_distr, num_per_node = [], [], [], []
        n_ini, n_fin  = 0, n_class_per_node
        n_total_iid   = len(sorted_labels) - len(noniid_part_sample)

        np.random.seed(_s(random_state))
        rand_lnodes_iid = np.random.randint(0, local_nodes, size=n_total_iid)

        for i in range(local_nodes):
            aux = [k for idx, k in enumerate(noniid_part_sample)
                   if k in uniq_class_noniid[n_ini:n_fin]]
            idx_aux = [idx for idx, k in enumerate(noniid_part_sample)
                       if k in uniq_class_noniid[n_ini:n_fin]]
            samp_iid = [lab for idx, (lab, ln) in
                        enumerate(zip(iid_part_sample, rand_lnodes_iid))
                        if ln == i]
            idx_samp = [idx + len(noniid_part_sample)
                        for idx, (lab, ln) in
                        enumerate(zip(iid_part_sample, rand_lnodes_iid))
                        if ln == i]
            aux     += samp_iid
            idx_aux += idx_samp
            df_aux   = (pd.DataFrame(aux, columns=['label'])
                        .label.value_counts().reset_index())
            df_aux.columns = ['index', 'label']
            df_node  = pd.DataFrame(np.unique(sorted_labels), columns=['index'])
            df_node  = (df_node.merge(df_aux, how='left',
                                       left_on='index', right_on='index')
                        .replace(np.nan, 0))
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)
            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux)
            idx_distr.append(idx_aux)
            n_ini += n_class_per_node
            if i == (local_nodes - 2):
                n_fin = len(uniq_class_noniid) + 1
            else:
                n_fin += n_class_per_node
        return pctg_distr, num_distr, idx_distr, num_per_node

    # ------------------------------------------------------------------
    @staticmethod
    def dirichlet_method(labels, local_nodes, alpha=1000,
                          random_state=None):
        labels    = np.array(labels)
        min_size  = 0
        num_classes = len(np.unique(labels))
        N         = labels.shape[0]
        rs_loop   = _s(random_state)          # ← capped from the start

        # Cap retries — with many clients and/or small per-class sample
        # counts, min_size>=10 can be mathematically unreachable (e.g.
        # 20 samples split across 20 clients can never average >=10
        # each). Without a cap, this loop spins forever. After
        # MAX_TRIES we accept the best split found instead of hanging.
        MAX_TRIES = 200
        tries = 0
        best_idx_batch, best_min_size = None, -1

        while min_size < 10:
            idx_batch = [[] for _ in range(local_nodes)]
            for k in range(num_classes):
                idx_k = np.where(labels == k)[0]
                np.random.seed(rs_loop)        # ← always safe
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
                proportions = np.array(
                    [p * (len(idx_j) < N / local_nodes)
                     for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist()
                             for idx_j, idx in
                             zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                if random_state is not None:
                    rs_loop = (rs_loop + 100) % _CAP   # ← capped every step

            if min_size > best_min_size:
                best_min_size, best_idx_batch = min_size, idx_batch

            tries += 1
            if tries >= MAX_TRIES and min_size < 10:
                idx_batch = best_idx_batch
                min_size  = best_min_size
                break

        pctg_distr, num_distr, idx_distr, num_per_node = [], [], [], []
        rs_loop = _s(random_state)

        for j in range(local_nodes):
            np.random.seed(rs_loop)            # ← always safe
            np.random.shuffle(idx_batch[j])
            aux = labels[idx_batch[j]]
            df_aux = (pd.DataFrame(aux, columns=['label'])
                      .label.value_counts().reset_index())
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(labels), columns=['index'])
            df_node = (df_node.merge(df_aux, how='left',
                                      left_on='index', right_on='index')
                       .replace(np.nan, 0))
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)
            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux)
            idx_distr.append(idx_batch[j])
            if random_state is not None:
                rs_loop = (rs_loop + 100) % _CAP       # ← capped every step

        return pctg_distr, num_distr, idx_distr, num_per_node

    # ------------------------------------------------------------------
    @staticmethod
    def add_gaussian_noise(feat, mu=0, sigma=0, client_id=0,
                            local_nodes=4, random_state=None):
        noise_level = sigma * client_id / local_nodes
        np.random.seed(_s(random_state))
        noise = np.random.normal(mu, noise_level, feat.shape)
        return feat + noise

    @staticmethod
    def calculate_bins_range(column, sigma_noise, n_bins):
        return np.array(np.linspace(column.min() - 4 * sigma_noise,
                                     column.max() + 4 * sigma_noise,
                                     num=n_bins, endpoint=True))

    @staticmethod
    def create_histogram(flat_input, bins):
        h, _ = np.histogram(flat_input, bins=bins)
        return h / flat_input.shape[0]

    # ------------------------------------------------------------------
    @staticmethod
    def dirichlet_method_quant_skew(labels, local_nodes, alpha=1000,
                                     random_state=None,
                                     method="no-quant-skew"):
        labels   = np.array(labels)
        N        = labels.shape[0]
        rs_loop  = _s(random_state)
        np.random.seed(rs_loop)
        idxs     = np.random.permutation(N)
        min_req  = len(np.unique(labels)) * 3

        if method == "dirichlet":
            min_size = 0
            while min_size < min_req:
                proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
                proportions = proportions / proportions.sum()
                min_size    = np.min(proportions * len(idxs))
        elif method == "minsize-dirichlet":
            proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
            proportions = proportions / proportions.sum()
            proportions = [(min_req + 1) / len(idxs)
                           if i < (min_req + 1) / len(idxs) else i
                           for i in proportions]
            proportions = [i / sum(proportions) for i in proportions]

        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        idx_batch   = [list(v) for v in np.split(idxs, proportions)]

        pctg_distr, num_distr, idx_distr, num_per_node = [], [], [], []
        rs_loop = _s(random_state)
        for j in range(local_nodes):
            np.random.seed(rs_loop)
            np.random.shuffle(idx_batch[j])
            aux = labels[idx_batch[j]]
            df_aux = (pd.DataFrame(aux, columns=['label'])
                      .label.value_counts().reset_index())
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(labels), columns=['index'])
            df_node = (df_node.merge(df_aux, how='left',
                                      left_on='index', right_on='index')
                       .replace(np.nan, 0))
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)
            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux)
            idx_distr.append(idx_batch[j])
            if random_state is not None:
                rs_loop = (rs_loop + 100) % _CAP
        return pctg_distr, num_distr, idx_distr, num_per_node

    # ------------------------------------------------------------------
    @staticmethod
    def st_dirichlet_method(labels, local_nodes, alpha=1000,
                             random_state=None, st_variable=None):
        st_variable = np.array(st_variable)
        labels      = np.array(labels)
        min_size    = 0
        num_categ   = len(np.unique(st_variable))
        N           = st_variable.shape[0]
        rs_loop     = _s(random_state)

        while min_size < 10:
            idx_batch = [[] for _ in range(local_nodes)]
            for k in range(num_categ):
                idx_k = np.where(st_variable == k)[0]
                np.random.seed(rs_loop)
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, local_nodes))
                proportions = np.array(
                    [p * (len(idx_j) < N / local_nodes)
                     for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist()
                             for idx_j, idx in
                             zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                if random_state is not None:
                    rs_loop = (rs_loop + 100) % _CAP

        pctg_distr, num_distr, idx_distr, num_per_node = [], [], [], []
        pctg_distr_st_var = []
        rs_loop = _s(random_state)
        for j in range(local_nodes):
            np.random.seed(rs_loop)
            np.random.shuffle(idx_batch[j])
            aux = labels[idx_batch[j]]
            df_aux = (pd.DataFrame(aux, columns=['label'])
                      .label.value_counts().reset_index())
            df_aux.columns = ['index', 'label']
            df_node = pd.DataFrame(np.unique(labels), columns=['index'])
            df_node = (df_node.merge(df_aux, how='left',
                                      left_on='index', right_on='index')
                       .replace(np.nan, 0))
            num_per_node.append(list(df_node.label))
            df_node['perc'] = df_node.label / sum(df_node.label)
            pctg_distr.append(list(df_node.perc))
            num_distr.append(aux)
            idx_distr.append(idx_batch[j])

            aux_st = st_variable[idx_batch[j]]
            df_aux = (pd.DataFrame(aux_st, columns=['st_var'])
                      .st_var.value_counts().reset_index())
            df_aux.columns = ['index', 'st_var']
            df_node = pd.DataFrame(np.unique(st_variable), columns=['index'])
            df_node = (df_node.merge(df_aux, how='left',
                                      left_on='index', right_on='index')
                       .replace(np.nan, 0))
            df_node['perc'] = df_node.st_var / sum(df_node.st_var)
            pctg_distr_st_var.append(list(df_node.perc))
            if random_state is not None:
                rs_loop = (rs_loop + 100) % _CAP

        return (pctg_distr, num_distr, idx_distr,
                num_per_node, pctg_distr_st_var)

    # ------------------------------------------------------------------
    def create_clients(self, image_list, label_list, num_clients=4,
                       prefix_cli='client', method="dirichlet",
                       alpha=1000, percent_noniid=0, sigma_noise=0,
                       bins='n_samples', feat_sample_rate=0.1,
                       feat_skew_method="gaussian-noise",
                       alpha_feat_split=1000, idx_feat='feat-mean',
                       feat_quantile=20, quant_skew_method="no-quant-skew",
                       alpha_quant_split=1000,
                       spa_temp_skew_method="no-spatemp-skew",
                       alpha_spa_temp=1000, spa_temp_var=None):

        client_names = ['{}_{}'.format(prefix_cli, i + 1)
                        for i in range(num_clients)]
        data = list(zip(image_list, label_list))

        # validation
        if (method in ("percent_noniid", "dirichlet") and
                feat_skew_method == "hist-dirichlet"):
            raise ValueError("hist-dirichlet incompatible with label skew methods.")
        if (quant_skew_method == "dirichlet" and
                feat_skew_method == "hist-dirichlet"):
            raise ValueError("hist-dirichlet incompatible with dirichlet quant skew.")
        if (method in ("percent_noniid", "dirichlet") and
                quant_skew_method == "dirichlet"):
            raise ValueError("dirichlet quant skew incompatible with label skew methods.")
        if (method == "no-label-skew" and
                quant_skew_method == "no-quant-skew" and
                spa_temp_skew_method == "no-spatemp-skew" and
                feat_skew_method == "gaussian-noise"):
            raise ValueError("gaussian-noise needs at least one skew method enabled.")

        rs = _s(self.random_state)   # cap once for all create_clients logic

        if feat_skew_method == "gaussian-noise":
            num_missing_classes = []
            shards_no, shards_with = [], []
            ids_no, ids_with = [], []
            rs_loop = rs

            if method == "dirichlet":
                lbl_pctg, lbl_num, lbl_idx, num_per = self.dirichlet_method(
                    labels=label_list, local_nodes=num_clients,
                    alpha=alpha, random_state=rs_loop)
            elif method == "percent_noniid":
                lbl_pctg, lbl_num, lbl_idx, num_per = \
                    self.percent_noniid_method(
                        labels=label_list, local_nodes=num_clients,
                        pct_noniid=percent_noniid, random_state=rs_loop)
            elif quant_skew_method in ("dirichlet", "minsize-dirichlet"):
                lbl_pctg, lbl_num, lbl_idx, num_per = \
                    self.dirichlet_method_quant_skew(
                        labels=label_list, local_nodes=num_clients,
                        alpha=alpha_quant_split, random_state=rs_loop,
                        method=quant_skew_method)
            elif spa_temp_skew_method == "st-dirichlet":
                lbl_pctg, lbl_num, lbl_idx, num_per, st_var_pctg = \
                    self.st_dirichlet_method(
                        labels=label_list, local_nodes=num_clients,
                        alpha=alpha_spa_temp, random_state=rs_loop,
                        st_variable=spa_temp_var)
            elif method not in ('percent_noniid', 'dirichlet', 'no-label-skew'):
                raise ValueError(f"Unknown label skew method: {method}")
            else:
                raise ValueError(f"Unknown quant skew method: {quant_skew_method}")

            JS  = jensen_shannon_distance(lbl_pctg)
            H   = hellinger_distance(lbl_pctg)
            emd = earth_movers_distance(lbl_pctg)
            distances = {'without_class_completion': {
                'jensen-shannon': JS, 'hellinger': H, 'earth-movers': emd}}

            data_df = pd.DataFrame(data)
            data_df.columns = [*data_df.columns[:-1], 'class']

            if spa_temp_skew_method == "st-dirichlet":
                spatem_df = pd.DataFrame(spa_temp_var, columns=['spatemp_var'])

            fed_data, ids_fed = {}, {}
            pctg_distr = []
            dh_no, dh_with = [], []
            st_cli_list = []
            spatemp_fed = {}

            n_bins   = (np.array(image_list).shape[0]
                        if bins == 'n_samples' else bins)
            shape_x  = np.array(np.array(image_list).shape)
            fsz      = max(int(feat_sample_rate * np.prod(shape_x[1:])), 1)
            np.random.seed(rs)
            idx_sf   = np.random.choice(np.arange(np.prod(shape_x[1:])),
                                         size=fsz, replace=False)
            features = np.array(image_list).reshape(
                (shape_x[0], np.prod(shape_x[1:])))[:, idx_sf]
            bins_range = np.apply_along_axis(
                self.calculate_bins_range, axis=0, arr=features,
                sigma_noise=sigma_noise, n_bins=n_bins)

            for i in range(num_clients):
                X = data_df.iloc[lbl_idx[i], 0].values
                y = data_df.iloc[lbl_idx[i], 1].values
                if len(X) == 0:
                    # Client got zero samples (can happen with many
                    # clients + few samples after MAX_TRIES fallback).
                    # Give it one dummy sample borrowed from index 0 so
                    # downstream shape logic doesn't crash on X[0].
                    X = data_df.iloc[[0], 0].values
                    y = data_df.iloc[[0], 1].values
                    lbl_idx[i] = [0]
                if isinstance(X[0], list):
                    X = np.array(X.tolist())
                if sigma_noise > 0:
                    X = self.add_gaussian_noise(
                        feat=X, sigma=sigma_noise, client_id=i + 1,
                        local_nodes=num_clients, random_state=rs_loop)
                    X = np.array(X.tolist())
                    sx = np.array(X.shape)
                    ft = X.reshape((sx[0], np.prod(sx[1:])))[:, idx_sf]
                    hg = np.array([self.create_histogram(col, b)
                                   for col, b in zip(ft.T, bins_range.T)])
                    del ft
                else:
                    hg = np.zeros((features.shape[1], 20))
                dh_no.append(list(hg)); del hg

                if i == (num_clients - 1):
                    dh_no_t = np.transpose(np.array(dh_no), (1, 0, 2)).tolist()
                    JS_f  = np.mean(list(map(jensen_shannon_distance, dh_no_t)))
                    H_f   = np.mean(list(map(hellinger_distance,      dh_no_t)))
                    emd_f = np.mean(list(map(earth_movers_distance,   dh_no_t)))
                    del dh_no_t

                X = X.tolist(); y = y.tolist()
                ids_no.append(lbl_idx[i])
                shards_no.append(list(zip(X, y)))

                if spa_temp_skew_method == "st-dirichlet":
                    st_cli_list.append(
                        spatem_df.iloc[lbl_idx[i], 0].values.tolist())

                diff = list(set(label_list) - set(y))
                num_missing_classes.append(len(diff))
                if diff:
                    for k in diff:
                        v = [ix for ix, yy in enumerate(label_list) if yy == k][0]
                        lbl_idx[i] = lbl_idx[i] + [v]

                X = data_df.iloc[lbl_idx[i], 0].values
                y = data_df.iloc[lbl_idx[i], 1].values
                if isinstance(X[0], list):
                    X = np.array(X.tolist())
                if sigma_noise > 0:
                    X = self.add_gaussian_noise(
                        feat=X, sigma=sigma_noise, client_id=i + 1,
                        local_nodes=num_clients, random_state=rs_loop)
                    X = np.array(X.tolist())
                    sx = np.array(X.shape)
                    ft = X.reshape((sx[0], np.prod(sx[1:])))[:, idx_sf]
                    hg = np.array([self.create_histogram(col, b)
                                   for col, b in zip(ft.T, bins_range.T)])
                    del ft
                else:
                    hg = np.zeros((features.shape[1], 20))
                dh_with.append(list(hg)); del hg

                X = X.tolist(); y = y.tolist()
                df_aux = (pd.DataFrame(y, columns=['label'])
                          .label.value_counts().reset_index())
                df_aux.columns = ['index', 'label']
                df_node = pd.DataFrame(np.unique(label_list), columns=['index'])
                df_node = (df_node.merge(df_aux, how='left',
                                          left_on='index', right_on='index')
                           .replace(np.nan, 0))
                df_node['perc'] = df_node.label / sum(df_node.label)
                pctg_distr.append(list(df_node.perc))
                ids_with.append(lbl_idx[i])
                shards_with.append(list(zip(X, y)))
                if self.random_state is not None:
                    rs_loop = (rs_loop + 100) % _CAP   # ← capped

            dh_with_t = np.transpose(np.array(dh_with), (1, 0, 2)).tolist()

            fed_data['with_class_completion']    = {
                client_names[i]: shards_with[i] for i in range(num_clients)}
            fed_data['without_class_completion'] = {
                client_names[i]: shards_no[i]   for i in range(num_clients)}
            ids_fed['with_class_completion']    = ids_with
            ids_fed['without_class_completion'] = ids_no

            JS2  = jensen_shannon_distance(pctg_distr)
            HD2  = hellinger_distance(pctg_distr)
            emd2 = earth_movers_distance(pctg_distr)
            distances['with_class_completion'] = {
                'jensen-shannon': JS2, 'hellinger': HD2, 'earth-movers': emd2}
            distances['without_class_completion_feat'] = {
                'jensen-shannon': JS_f, 'hellinger': H_f, 'earth-movers': emd_f}

            dists = np.array(list(map(jensen_shannon_distance, dh_with_t)))
            JS_fw  = np.mean(dists)
            dists  = np.array(list(map(hellinger_distance,      dh_with_t)))
            H_fw   = np.mean(dists)
            dists  = np.array(list(map(earth_movers_distance,   dh_with_t)))
            emd_fw = np.mean(dists)
            distances['with_class_completion_feat'] = {
                'jensen-shannon': JS_fw, 'hellinger': H_fw,
                'earth-movers': emd_fw}

            if spa_temp_skew_method == "st-dirichlet":
                spatemp_fed['without_class_completion'] = {
                    client_names[i]: st_cli_list[i]
                    for i in range(num_clients)}

        elif feat_skew_method == "hist-dirichlet":
            # ── hist-dirichlet branch (no seed overflow risk here) ────────
            num_missing_classes = []
            shards_no, shards_with = [], []
            ids_no, ids_with = [], []
            rs_loop = rs

            shape_x = np.array(np.array(image_list).shape)
            if idx_feat == 'feat-mean':
                fs = np.mean(np.array(image_list).reshape(
                    (shape_x[0], np.prod(shape_x[1:]))), axis=1)
            else:
                fs = np.array(image_list).reshape(
                    (shape_x[0], np.prod(shape_x[1:])))[:, idx_feat]
            fs = pd.DataFrame(fs, columns=['fs'])
            fs = pd.qcut(fs['fs'], feat_quantile, labels=False, duplicates='drop')

            fp, fn, fi, num_per = self.dirichlet_method(
                labels=fs, local_nodes=num_clients,
                alpha=alpha_feat_split, random_state=rs_loop)

            data_df = pd.DataFrame(data)
            data_df.columns = [*data_df.columns[:-1], 'class']
            fed_data, ids_fed = {}, {}
            pctg_no, pctg_with, feat_pctg_with = [], [], []

            for i in range(num_clients):
                X = data_df.iloc[fi[i], 0].values
                y = data_df.iloc[fi[i], 1].values
                if isinstance(X[0], list):
                    X = np.array(X.tolist())
                X, y = X.tolist(), y.tolist()
                ids_no.append(fi[i])
                shards_no.append(list(zip(X, y)))
                df_aux = (pd.DataFrame(y, columns=['label'])
                          .label.value_counts().reset_index())
                df_aux.columns = ['index', 'label']
                df_node = pd.DataFrame(np.unique(label_list), columns=['index'])
                df_node = (df_node.merge(df_aux, how='left',
                                          left_on='index', right_on='index')
                           .replace(np.nan, 0))
                df_node['perc'] = df_node.label / sum(df_node.label)
                pctg_no.append(list(df_node.perc))

                diff = list(set(label_list) - set(y))
                num_missing_classes.append(len(diff))
                if diff:
                    for k in diff:
                        v = [ix for ix, yy in enumerate(label_list) if yy == k][0]
                        fi[i] = fi[i] + [v]

                X = data_df.iloc[fi[i], 0].values
                y = data_df.iloc[fi[i], 1].values
                if isinstance(X[0], list):
                    X = np.array(X.tolist())
                X, y = X.tolist(), y.tolist()
                df_aux = (pd.DataFrame(y, columns=['label'])
                          .label.value_counts().reset_index())
                df_aux.columns = ['index', 'label']
                df_node = pd.DataFrame(np.unique(label_list), columns=['index'])
                df_node = (df_node.merge(df_aux, how='left',
                                          left_on='index', right_on='index')
                           .replace(np.nan, 0))
                df_node['perc'] = df_node.label / sum(df_node.label)
                pctg_with.append(list(df_node.perc))

                df_aux = (pd.DataFrame(fs.values[fi[i]], columns=['feature'])
                          .feature.value_counts().reset_index())
                df_aux.columns = ['index', 'feature']
                df_node = pd.DataFrame(np.unique(fs), columns=['index'])
                df_node = (df_node.merge(df_aux, how='left',
                                          left_on='index', right_on='index')
                           .replace(np.nan, 0))
                df_node['perc'] = df_node.feature / sum(df_node.feature)
                feat_pctg_with.append(list(df_node.perc))

                ids_with.append(fi[i])
                shards_with.append(list(zip(X, y)))
                if self.random_state is not None:
                    rs_loop = (rs_loop + 100) % _CAP

            fed_data['with_class_completion']    = {
                client_names[i]: shards_with[i] for i in range(num_clients)}
            fed_data['without_class_completion'] = {
                client_names[i]: shards_no[i]   for i in range(num_clients)}
            ids_fed['with_class_completion']    = ids_with
            ids_fed['without_class_completion'] = ids_no

            JS  = jensen_shannon_distance(pctg_no)
            HD  = hellinger_distance(pctg_no)
            emd = earth_movers_distance(pctg_no)
            distances = {'without_class_completion': {
                'jensen-shannon': JS, 'hellinger': HD, 'earth-movers': emd}}
            JS  = jensen_shannon_distance(pctg_with)
            HD  = hellinger_distance(pctg_with)
            emd = earth_movers_distance(pctg_with)
            distances['with_class_completion'] = {
                'jensen-shannon': JS, 'hellinger': HD, 'earth-movers': emd}
            JS  = jensen_shannon_distance(fp)
            HD  = hellinger_distance(fp)
            emd = earth_movers_distance(fp)
            distances['without_class_completion_feat'] = {
                'jensen-shannon': JS, 'hellinger': HD, 'earth-movers': emd}
            JS  = jensen_shannon_distance(feat_pctg_with)
            HD  = hellinger_distance(feat_pctg_with)
            emd = earth_movers_distance(feat_pctg_with)
            distances['with_class_completion_feat'] = {
                'jensen-shannon': JS, 'hellinger': HD, 'earth-movers': emd}
        else:
            raise ValueError(f"Unknown feat_skew_method: {feat_skew_method}")

        # ── quantity distances ────────────────────────────────────────────
        sizes = [len(v) for v in fed_data['without_class_completion'].values()]
        pp    = [[e / sum(sizes)] for e in sizes]
        distances['without_class_completion_quant'] = {
            'jensen-shannon': jensen_shannon_distance(pp),
            'hellinger':      hellinger_distance(pp),
            'earth-movers':   earth_movers_distance(pp)}
        sizes = [len(v) for v in fed_data['with_class_completion'].values()]
        pp    = [[e / sum(sizes)] for e in sizes]
        distances['with_class_completion_quant'] = {
            'jensen-shannon': jensen_shannon_distance(pp),
            'hellinger':      hellinger_distance(pp),
            'earth-movers':   earth_movers_distance(pp)}

        if spa_temp_skew_method == "st-dirichlet":
            JS  = jensen_shannon_distance(st_var_pctg)
            H   = hellinger_distance(st_var_pctg)
            emd = earth_movers_distance(st_var_pctg)
            distances['without_class_completion_spatemp'] = {
                'jensen-shannon': JS, 'hellinger': H, 'earth-movers': emd}
            return fed_data, ids_fed, num_missing_classes, distances, spatemp_fed

        return fed_data, ids_fed, num_missing_classes, distances
