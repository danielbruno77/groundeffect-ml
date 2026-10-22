import numpy as np
import pytest

from groundeffect.windtunnel.detect import cusum, datum_chart, hampel, pelt, run_qa, score
from groundeffect.windtunnel.synth import simulate_campaign

KINDS = {"spike": "spike", "drift": "drift_or_step", "step": "drift_or_step", "tap_blocked": "tap", "config_outlier": "config_outlier"}


def test_hampel_flags_isolated_spike():
    x = np.random.default_rng(0).normal(0, 0.01, 200)
    x[100] += 0.2
    flags = hampel(x, sigma=0.01)
    assert flags[100] and flags.sum() <= 2


def test_cusum_detects_small_persistent_shift():
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, 1, 50), rng.normal(1.0, 1, 50)]
    _, _, alarm = cusum(x, 1.0)
    first = int(np.argmax(alarm))
    assert alarm.any() and 50 <= first < 70


def test_datum_chart_resets_after_alarm():
    dev = np.r_[np.zeros(5), 0.05, np.zeros(10)]
    chart = datum_chart(dev, sigma=0.005)
    assert chart["alarm"][5] and not chart["alarm"][7:].any()


def test_pelt_finds_mean_shift():
    rng = np.random.default_rng(2)
    x = np.r_[rng.normal(0, 0.1, 40), rng.normal(1, 0.1, 40)]
    cps = pelt(x, sigma=0.1)
    assert any(abs(c - 40) <= 1 for c in cps)


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_campaign_qa_detects_every_injected_fault(seed):
    camp = simulate_campaign(seed=seed)
    qa = run_qa(camp.samples, camp.taps)
    for kind, col in KINDS.items():
        truth = np.zeros(len(camp.samples), dtype=bool)
        for f in camp.faults:
            if KINDS[f.kind] == col:
                truth[f.start : f.end + 1] = True
        s = score(qa.flags[col].to_numpy(), truth, [(f.start, f.end) for f in camp.faults if f.kind == kind])
        assert s["recall"] == 1.0, kind
        assert s["precision"] > 0.7, kind


def test_correlation_absorbs_cfd_tunnel_offset():
    camp = simulate_campaign(correlation=(0.9, 0.05))
    qa = run_qa(camp.samples, camp.taps)
    assert qa.correlation.k == pytest.approx(0.9, abs=0.02)
    assert qa.correlation.b == pytest.approx(0.05, abs=0.02)


def test_ewma_false_alarm_rate_matches_design():
    # In-control data: with L = 3 the per-point false alarm rate must be small,
    # including right after a restart (the bug class this guards against).
    rng = np.random.default_rng(5)
    alarms = [datum_chart(rng.normal(0, 1, 40), sigma=1.0, h=1e9, shewhart=1e9)["alarm"].mean() for _ in range(200)]
    assert np.mean(alarms) < 0.01
