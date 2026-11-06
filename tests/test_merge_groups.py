import torch  # noqa: F401
import warnings

import numpy as np
import pandas as pd

from auto_inspector import assign_merge_groups, metrics_to_decision


def _old_merge_step(metrics_df, FBD, pxlthr_distance_boundary, d_snr_thr):
    """Copy of the merge step of auto_inspector.py at commit 63dfe9c (the reference).

    Two deviations from the verbatim copy, and only these: the sort is stable and reassigned instead of
    inplace. The old unstable sort left the visit order of the members of one cluster (all tied on
    corr_groups) to the NumPy build, and the SNR-loser rule depends on that order, so the reference
    needs the same fixed order to be comparable.
    """
    metrics_df["merge"] = 0
    # unique number of clusters
    unique_clusters = metrics_df.loc[metrics_df['corr_groups'] != 0, 'corr_groups'].unique()
    unique_clusters = sorted(unique_clusters, reverse=True)

    metrics_df = metrics_df.sort_values('corr_groups', ascending=False, kind='stable')
    for cluster_id in unique_clusters:
        # indeces of neurons in cluster
        cluster = metrics_df[metrics_df['corr_groups'] == cluster_id]
        cluster_indices = cluster['component_idx'].tolist()

        for index_1 in cluster_indices:
            for index_2 in cluster_indices:
                # address of current neuron
                metrics_df_mask_1 = metrics_df['component_idx'] == index_1
                metrics_df_mask_2 = metrics_df['component_idx'] == index_2

                # execute only if not already classified to delete
                if (index_1 != index_2 and metrics_df.loc[metrics_df_mask_1, 'delete'].item() == 0
                        and metrics_df.loc[metrics_df_mask_2, 'delete'].item() == 0):

                    # pandas df indeces matching with FBD matrix
                    FBD_idx_1 = metrics_df.loc[metrics_df_mask_1].index[0]
                    FBD_idx_2 = metrics_df.loc[metrics_df_mask_2].index[0]

                    # SNRs of current neurons
                    snr = [metrics_df.loc[metrics_df_mask_1, 'caiman_snr'].item(),
                           metrics_df.loc[metrics_df_mask_2, 'caiman_snr'].item()]

                    # check for merge/delete/keep choice depending on rules
                    if FBD[FBD_idx_1][FBD_idx_2] <= pxlthr_distance_boundary:
                        d_snr = max(snr) - min(snr)
                        # merge neurons with close snr value
                        if d_snr <= d_snr_thr:
                            metrics_df.loc[metrics_df_mask_1, 'merge'] = cluster_id

                        # delete all with smaller SNR
                        elif snr[0] != max(snr):
                            metrics_df.loc[metrics_df_mask_1, 'delete'] = 1
                    # else : default value is 0 in 'delete' if it has not been classified already

    # Clean up conflicts: if neuron is marked for deletion, remove from merge groups
    # Deletion is primary - no further actions allowed on deleted neurons
    deleted_neurons = metrics_df['delete'] == 1
    metrics_df.loc[deleted_neurons, 'merge'] = 0

    metrics_df = metrics_df.sort_index()
    return metrics_df


def _df(snr, groups, delete=None, ids=None):
    n = len(snr)
    return pd.DataFrame({"component_idx": ids if ids is not None else np.arange(n) * 2 + 1,
                         "corr_groups": groups, "caiman_snr": snr,
                         "delete": delete if delete is not None else [0] * n})


def _fbd(n, close):  # close: {(i, j): distance}; all other pairs inf, as for uncorrelated pairs
    m = np.full((n, n), np.inf)
    for (i, j), d in close.items():
        m[i, j] = m[j, i] = d
    return m


def _fbd_connected(fbd, rows, thr):
    sub = fbd[np.ix_(rows, rows)] <= thr
    seen, stack = {0}, [0]
    while stack:
        k = stack.pop()
        for j in np.nonzero(sub[k])[0]:
            if int(j) not in seen:
                seen.add(int(j))
                stack.append(int(j))
    return len(seen) == len(rows)


def test_two_far_pairs_in_one_cluster_are_two_groups():
    out = assign_merge_groups(_df([5.0] * 4, [1.0] * 4), _fbd(4, {(0, 1): 5, (2, 3): 5}))
    assert out["merge"].tolist() == [1, 1, 2, 2] and out["delete"].tolist() == [0] * 4


def test_chain_is_one_group():
    out = assign_merge_groups(_df([5.0] * 3, [1.0] * 3), _fbd(3, {(0, 1): 5, (1, 2): 5, (0, 2): 40}))
    assert out["merge"].tolist() == [1, 1, 1]


def test_member_deleted_later_drops_out():
    # C (row 2) is linked to B, then deleted as the dim copy of D (SNR 7 vs 30): A-B stay a group
    out = assign_merge_groups(_df([5.0, 6.0, 7.0, 30.0], [1.0] * 4),
                              _fbd(4, {(0, 1): 5, (1, 2): 5, (2, 3): 5}))
    assert out["delete"].tolist() == [0, 0, 1, 0] and out["merge"].tolist() == [1, 1, 0, 0]


def test_groups_are_numbered_across_clusters():
    out = assign_merge_groups(_df([5.0] * 4, [2.0, 2.0, 1.0, 1.0]), _fbd(4, {(0, 1): 5, (2, 3): 5}))
    assert out["merge"].tolist() == [1, 1, 2, 2]


def test_pair_at_the_cutoff():
    out = assign_merge_groups(_df([5.0, 15.0, 5.0, 5.0], [1.0] * 4), _fbd(4, {(0, 1): 15.0, (2, 3): 15.0001}))
    assert out["merge"].tolist() == [1, 1, 0, 0] and out["delete"].tolist() == [0] * 4


def test_single_component_has_no_group():
    out = assign_merge_groups(_df([3.0], [None]), np.full((1, 1), np.inf))
    assert out["merge"].tolist() == [0] and out["delete"].tolist() == [0]


def test_random_inputs_keep_old_deletes():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n = 40
        snr = rng.uniform(1, 30, n); snr[rng.random(n) < 0.1] = np.nan
        df = _df(snr, rng.integers(0, 4, n).astype(float), (rng.random(n) < 0.2).astype(int))
        fbd = rng.uniform(0, 40, (n, n)); fbd = np.minimum(fbd, fbd.T); np.fill_diagonal(fbd, np.inf)
        old = _old_merge_step(df.copy(), fbd, 15, 10)
        new = assign_merge_groups(df.copy(), fbd, 15, 10)
        assert new["delete"].tolist() == old["delete"].tolist(), seed
        for gid, g in new[new["merge"] > 0].groupby("merge"):
            assert len(g) >= 2 and (g["delete"] == 0).all()
            assert g["merge"].nunique() == 1 and old.loc[g.index, "merge"].nunique() == 1
            assert (old.loc[g.index, "merge"] != 0).all()     # inside one old group
            assert _fbd_connected(fbd, g.index.to_numpy(), 15)  # every new group is FBD-connected
        # every kept pair that passes the pair rule ends in one group
        kept = new.index[new["delete"] == 0]
        for i in kept:
            for j in kept:
                if (i < j and new.at[i, "corr_groups"] == new.at[j, "corr_groups"] != 0
                        and fbd[i, j] <= 15 and abs(snr[i] - snr[j]) <= 10):
                    assert new.at[i, "merge"] == new.at[j, "merge"] != 0


def test_metrics_to_decision_uses_the_new_groups():
    df = _df([5.0] * 4, [1.0] * 4).assign(area=10.0)
    out = metrics_to_decision(df, None, None, _fbd(4, {(0, 1): 5, (2, 3): 5}),
                              deletion_rules=["area<1"], brain="thresholds")
    assert out["merge"].tolist() == [1, 1, 2, 2]


def test_deletions_follow_row_order_not_memory_layout():
    """Chain A - B - C (A and C far apart): B is dimmer than A by more than d_snr_thr, C dimmer than B.
    Whether C survives depends on whether B is deleted before B visits C, i.e. on the visit order. The
    order is the frame's row order and does not change with dtype or memory layout."""
    fbd = _fbd(3, {(0, 1): 5, (1, 2): 5})
    df = _df([40.0, 25.0, 10.0], [1.0] * 3)

    def deleted(frame):
        out = assign_merge_groups(frame, fbd, 15, 10)
        return dict(zip(out.index.tolist(), out["delete"].tolist()))

    assert deleted(df) == {0: 0, 1: 1, 2: 0}               # B deleted by A first; C is never compared with A
    reversed_rows = df.iloc[::-1]
    assert deleted(reversed_rows) == {0: 0, 1: 1, 2: 1}    # C visits B first and is deleted
    for frame, expected in ((df, {0: 0, 1: 1, 2: 0}), (reversed_rows, {0: 0, 1: 1, 2: 1})):
        assert deleted(frame.copy()) == expected
        assert deleted(pd.concat([frame.iloc[:1], frame.iloc[1:]])) == expected
        assert deleted(frame.astype({"corr_groups": np.float32, "caiman_snr": np.float32})) == expected
        assert deleted(frame.astype({"component_idx": np.int32, "delete": np.int8})) == expected


def test_assign_merge_groups_leaves_the_input_and_warns_about_nothing():
    df = _df([5.0, 6.0], [1.0, 1.0])
    before = df.copy()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = assign_merge_groups(df, _fbd(2, {(0, 1): 5}), 15, 10)
    assert not [w for w in caught if issubclass(w.category, FutureWarning) or "Pandas" in w.category.__name__],         [str(w.message) for w in caught]
    pd.testing.assert_frame_equal(df, before)  # no 'merge' column added, no 'delete' change
    assert out["merge"].tolist() == [1, 1]
