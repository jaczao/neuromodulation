from prototype.data import make_sequence, task_indices

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
