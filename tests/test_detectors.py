import numpy as np
import pytest

from src import detectors as D


def test_gini_hand_cases():
    m = 10
    assert D.gini_durations([], m) == 0.0                 # S = 0
    assert D.gini_durations([4], m) == 0.0                # S = 1
    assert D.gini_durations(np.arange(1, 11), m) == 0.0   # S = m
    # t = (2, 5): d = (2, 3); sum|.| = 1; (1/4)*2*1/(2*2.5) = 0.1
    assert D.gini_durations([2, 5], m) == pytest.approx(0.1)
    # t = (1, 2, 6): d = (1, 1, 4); pairs: 0+3+3 = 6; (1/9)*2*6/(2*2) = 1/3
    assert D.gini_durations([1, 2, 6], m) == pytest.approx(1 / 3)
    # censored gap after the last violation is ignored: same as above in a longer window
    assert D.gini_durations([1, 2, 6], 50) == pytest.approx(1 / 3)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.5, 0.97])
def test_rolling_gini_matches_reference(p):
    rng = np.random.default_rng(1)
    I = (rng.random(400) < p).astype(np.int8)
    np.testing.assert_allclose(D.rolling_gini(I, 50), D.rolling_gini_reference(I, 50), rtol=0, atol=1e-14)


def test_rolling_gini_batched_shape():
    rng = np.random.default_rng(2)
    I = (rng.random((3, 4, 120)) < 0.1).astype(np.int8)
    out = D.rolling_gini(I, 30)
    assert out.shape == (3, 4, 91)
    np.testing.assert_array_equal(out[1, 2], D.rolling_gini(I[1, 2], 30))


def test_rolling_uc_uses_abs():
    I = np.zeros(20, np.int8)
    uc = D.rolling_uc(I, 10, 0.1)
    assert uc.shape == (11,)
    np.testing.assert_allclose(uc, 0.1)       # |0 - 0.1|, not -0.1
    I[:10] = 1
    assert D.rolling_uc(I, 10, 0.1)[0] == pytest.approx(0.9)


def test_null_moments_reproducible_and_sane():
    a = D.null_moments(250, 0.05, reps=5000, rng=3)
    b = D.null_moments(250, 0.05, reps=5000, rng=3)
    assert a == b
    # E|Binomial mean - p| roughly sqrt(2/pi) * sqrt(p(1-p)/m)
    assert a.mu_UC == pytest.approx(np.sqrt(2 / np.pi) * np.sqrt(0.05 * 0.95 / 250), rel=0.1)
