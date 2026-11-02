from src.promote import decide


def test_first_model_under_threshold_is_promoted():
    assert decide(8.0, None, max_rmse=12.0, min_improvement=0.0)[0] is True


def test_model_above_threshold_is_never_promoted():
    assert decide(13.0, None, max_rmse=12.0, min_improvement=0.0)[0] is False
    assert decide(13.0, 20.0, max_rmse=12.0, min_improvement=0.0)[0] is False


def test_candidate_must_beat_production():
    assert decide(7.0, 8.0, max_rmse=12.0, min_improvement=0.0)[0] is True
    assert decide(8.0, 8.0, max_rmse=12.0, min_improvement=0.0)[0] is False
    assert decide(7.9, 8.0, max_rmse=12.0, min_improvement=0.5)[0] is False
