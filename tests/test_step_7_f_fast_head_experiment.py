import csv

import pytest
import torch

from step_7_f_fast_head_experiment import cache_path_for, date_holdout_indices, fit_head, load_cached_feature, metrics, read_split, validate_feature_record


def test_head_training_uses_train_targets_without_mutating_initial_weights():
    features = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    labels = torch.tensor([0.6, 0.7, 0.8, 0.9])
    teacher = torch.full((4,), 0.5)
    initial_weight = torch.zeros(1)
    initial_bias = torch.tensor(0.0)
    result = fit_head((features, labels, teacher), (features, labels, teacher), initial_weight, initial_bias,
                      0.0, epochs=100, learning_rate=0.02, regularization=0.0)
    assert result["metrics"]["mae"] < metrics(teacher, labels, teacher)["mae"]
    assert torch.equal(initial_weight, torch.zeros(1)) and initial_bias.item() == 0


def test_split_validation_rejects_duplicates_and_unsafe_names(tmp_path):
    path = tmp_path / "split.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "waviness"])
        writer.writerows([["step-2_sea.jpg", "0.1"], ["step-2_sea.jpg", "0.2"]])
    with pytest.raises(ValueError, match="duplicate"):
        read_split(path)
    path.write_text("filename,waviness\n../step-2_sea.jpg,0.1\n")
    with pytest.raises(ValueError, match="Unexpected"):
        read_split(path)


def test_feature_cache_migrates_and_uses_current_labels(tmp_path):
    identity = {"filename": "step-2_sea.jpg", "source_sha256": "abc"}
    legacy = cache_path_for(tmp_path, {**identity, "label": "0.1"})
    torch.save({"features": torch.ones(2), "baseline": 0.2, "label": 0.1}, legacy)
    record, status = load_cached_feature(tmp_path, identity, "0.1")
    assert status == "cached (legacy)" and record["label"] == 0.1
    assert cache_path_for(tmp_path, identity).is_file()
    record, status = load_cached_feature(tmp_path, identity, "0.7")
    assert status == "cached" and record["label"] == 0.7


def test_feature_cache_reuses_verified_compatible_runner(tmp_path):
    previous = {"filename": "step-2_sea.jpg", "runner_hash": "previous"}
    current = {"filename": "step-2_sea.jpg", "runner_hash": "current"}
    torch.save({"features": torch.ones(2), "baseline": 0.2}, cache_path_for(tmp_path, previous))
    record, status = load_cached_feature(tmp_path, current, "0.4", compatible_identities=(previous,))
    assert status == "cached (compatible runner)" and record["label"] == 0.4
    assert cache_path_for(tmp_path, current).is_file()


def test_date_holdout_keeps_each_day_in_one_partition():
    days = ["day-a", "day-a", "day-b", "day-b", "day-c", "day-d"]
    fit, holdout = date_holdout_indices(days)
    assert {days[i] for i in fit}.isdisjoint({days[i] for i in holdout})
    assert len(fit) + len(holdout) == len(days)
    with pytest.raises(ValueError, match="two distinct"):
        date_holdout_indices(["same", "same"])


def test_invalid_cached_features_fail_before_training():
    with pytest.raises(RuntimeError, match="Invalid cached features"):
        validate_feature_record({"features": torch.tensor([float("nan")]), "baseline": .5}, 1, "sea.jpg")
    with pytest.raises(RuntimeError, match="Invalid cached features"):
        validate_feature_record({"features": torch.ones(1), "baseline": 2.0}, 1, "sea.jpg")
