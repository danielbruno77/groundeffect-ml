import numpy as np

from groundeffect.doe import DesignSpace, latin_hypercube, sobol


def test_unit_roundtrip_with_log_ride_height():
    space = DesignSpace()
    x = sobol(space, 64)
    assert np.allclose(space.from_unit(space.to_unit(x)), x)


def test_samples_within_bounds():
    space = DesignSpace()
    b = space.bounds()
    for x in (sobol(space, 128), latin_hypercube(space, 32)):
        assert np.all(x >= b[:, 0] - 1e-12) and np.all(x <= b[:, 1] + 1e-12)


def test_log_spacing_puts_more_points_at_low_ride_height():
    x = sobol(DesignSpace(), 256)
    mid = 0.5 * (0.08 + 0.6)
    assert (x[:, 0] < mid).mean() > 0.6
