import numpy as np
import pytest

from groundeffect.geometry import WingDesign, cp_stations, naca4, place_wing

BASE = WingDesign(h_c=0.15, alpha_deg=4.0, camber=0.04, thickness=0.12)


def test_naca_closed_te_and_thickness():
    upper, lower = naca4(BASE, 200)
    assert np.allclose(upper[-1], lower[-1])
    thickness = np.interp(0.3, upper[:, 0], upper[:, 1]) - np.interp(0.3, lower[:, 0], lower[:, 1])
    assert thickness == pytest.approx(0.12, rel=0.03)


@pytest.mark.parametrize("h_c", [0.08, 0.2, 0.6])
def test_ride_height_is_gap_to_lowest_point(h_c):
    w = place_wing(WingDesign(h_c, 5.0, 0.05, 0.1), chord=0.3)
    lowest = min(w.suction[:, 1].min(), w.pressure[:, 1].min())
    assert lowest == pytest.approx(h_c * 0.3, abs=1e-12)


def test_incidence_is_nose_down():
    w = place_wing(BASE)
    assert w.le[1] < w.te[1]  # LE lower than TE: downforce attitude
    assert np.degrees(np.arctan2(w.te[1] - w.le[1], w.te[0] - w.le[0])) == pytest.approx(4.0, abs=0.05)


def test_suction_side_faces_ground():
    w = place_wing(BASE)
    x = np.linspace(0.05, 0.9, 20) * w.chord
    ys = np.interp(x, w.suction[:, 0], w.suction[:, 1])
    yp = np.interp(x, w.pressure[:, 0], w.pressure[:, 1])
    assert np.all(ys < yp)


def test_to_local_inverts_placement():
    upper, _ = naca4(BASE)
    w = place_wing(BASE)
    assert np.allclose(w.to_local(w.suction), upper, atol=1e-12)


def test_stations_are_interior_and_sorted():
    st = cp_stations(80)
    assert st.min() > 0 and st.max() < 1 and np.all(np.diff(st) > 0)
