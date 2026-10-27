from step_7_h_benchmark_arrival import percentile95


def test_arrival_percentile_uses_nearest_rank():
    assert percentile95([.2]) == .2
    assert percentile95(list(range(1, 21))) == 19
