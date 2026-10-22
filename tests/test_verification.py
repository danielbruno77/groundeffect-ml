import pytest

from groundeffect.verification import gci


def test_gci_recovers_order_and_extrapolated_value():
    # phi = phi0 + C h^2 on three grids with refinement ratio sqrt(2) in h.
    n = (40_000, 20_000, 10_000)
    phi = tuple(1.3 + 50.0 / ni for ni in n)  # h^2 ~ 1/N in 2D
    g = gci(phi, n)
    assert g.order == pytest.approx(2.0, abs=1e-6)
    assert g.phi_extrapolated == pytest.approx(1.3, abs=1e-9)
    assert not g.oscillatory
    assert g.asymptotic_ratio == pytest.approx(1.0, rel=1e-2)


def test_gci_flags_oscillatory_convergence():
    g = gci((1.20, 1.25, 1.19), (40_000, 20_000, 10_000))
    assert g.oscillatory


def test_identical_solutions_raise():
    with pytest.raises(ValueError):
        gci((1.0, 1.0, 1.1), (4, 2, 1))
