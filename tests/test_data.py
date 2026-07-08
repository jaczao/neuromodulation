from prototype.data import SplitMNIST, make_sequence, task_indices

N_TASKS = 5
_SEED = 42


def test_each_task_has_two_classes():
    sequence = make_sequence(_SEED)
    for task_id, pair in enumerate(sequence):
        assert len(pair) == 2, f"Task {task_id} has {len(pair)} classes, expected 2"


def test_pairwise_class_sets_disjoint():
    sequence = make_sequence(_SEED)
    for i in range(N_TASKS):
        for j in range(i + 1, N_TASKS):
            shared = set(sequence[i]) & set(sequence[j])
            assert not shared, f"Tasks {i} and {j} share classes: {shared}"


def test_union_covers_all_classes():
    sequence = make_sequence(_SEED)
    union = set()
    for pair in sequence:
        union |= set(pair)
    assert union == set(range(10)), f"Union of task classes is {union}, expected {{0,...,9}}"


def test_no_train_test_index_leakage():
    """Within each MNIST split, every sample belongs to exactly one task (disjoint index sets)."""
    sequence = make_sequence(_SEED)

    train_index_sets = [set(task_indices(tid, "train", sequence)) for tid in range(N_TASKS)]
    test_index_sets = [set(task_indices(tid, "test", sequence)) for tid in range(N_TASKS)]

    for i in range(N_TASKS):
        for j in range(i + 1, N_TASKS):
            overlap_train = train_index_sets[i] & train_index_sets[j]
            assert not overlap_train, f"Tasks {i} and {j} share {len(overlap_train)} train indices"

            overlap_test = test_index_sets[i] & test_index_sets[j]
            assert not overlap_test, f"Tasks {i} and {j} share {len(overlap_test)} test indices"


def test_cl_val_split_disjoint_from_train_and_from_test():
    """CL val split partitions the task's TRAIN indices and is drawn from the TRAIN dataset."""
    ds = SplitMNIST(sequence=make_sequence(_SEED), val_frac=0.1)
    for tid in range(N_TASKS):
        train_idx, val_idx = ds._task_train_val_idx(tid)
        train_set, val_set = set(train_idx), set(val_idx)

        assert val_set, f"Task {tid} produced an empty val split at val_frac=0.1"
        # train/val are a disjoint partition of the full task train index set (no leakage, no loss)
        assert not (train_set & val_set), f"Task {tid} train and val indices overlap"
        full = set(task_indices(tid, "train", ds.sequence))
        assert train_set | val_set == full, f"Task {tid} train+val does not reconstruct the task train set"
        # ~10% held out
        assert abs(len(val_set) / len(full) - 0.1) < 0.01, f"Task {tid} val fraction off target"
        # the val LOADER is built from the TRAIN dataset object, so test-set leakage is impossible
        val_loader = ds.get_task_val_loader(tid)
        assert val_loader.dataset.dataset is ds._train_ds, f"Task {tid} val loader is not carved from TRAIN"


def test_cl_val_split_default_off_is_backwards_compatible():
    """val_frac=0 (the default) carves nothing → train unchanged, val loader unavailable."""
    ds = SplitMNIST(sequence=make_sequence(_SEED))  # val_frac defaults to 0.0
    for tid in range(N_TASKS):
        train_idx, val_idx = ds._task_train_val_idx(tid)
        assert not val_idx, f"Task {tid} unexpectedly carved a val split at val_frac=0"
        assert set(train_idx) == set(task_indices(tid, "train", ds.sequence))


def test_cl_val_split_is_deterministic_across_orderings():
    """Held-out images depend on the class pair, not the task order → stable across sequences."""
    ds_a = SplitMNIST(sequence=make_sequence(1), val_frac=0.1)
    ds_b = SplitMNIST(sequence=make_sequence(2), val_frac=0.1)
    # collect val indices keyed by class pair, independent of which task slot they land in
    val_a = {tuple(sorted(p)): set(ds_a._task_train_val_idx(t)[1]) for t, p in enumerate(ds_a.sequence)}
    val_b = {tuple(sorted(p)): set(ds_b._task_train_val_idx(t)[1]) for t, p in enumerate(ds_b.sequence)}
    for pair, idx_a in val_a.items():
        assert idx_a == val_b[pair], f"Val split for pair {pair} changed with task ordering"
