"""Physical checks of the circular backbone, including its graph distances."""
import numpy as np
import pytest

from lefs_cython.simple import LEFSimulator, constants


def ring_at(positions, capture=None, release=None):
    """Place specified motors on a 100-site ring with no random loss or pauses."""
    n = 100  # A small ring makes crossings and graph distances easy to inspect.
    sim = LEFSimulator(
        len(positions), n, np.ones(n), np.zeros((n, 5)),
        np.zeros((n, 2)) if capture is None else capture,
        np.zeros(n) if release is None else release, np.zeros(n),
        skip_load=True, circular=True,
    )
    sim.force_load_LEFs(np.asarray(positions, dtype=np.int32).reshape(-1, 2))
    return sim


@pytest.mark.parametrize("initial, expected", [([[0, 10]], [[99, 11]]),
                                              ([[10, 99]], [[9, 0]])])
def test_both_movement_directions_cross_zero(initial, expected):
    sim = ring_at(initial)
    sim.steps(0, 1)
    np.testing.assert_array_equal(sim.get_LEFs(), expected)


def test_heads_cannot_pass_across_zero_or_through_their_own_partner():
    sim = ring_at([[0, 20], [80, 99]])
    sim.steps(0, 1)
    np.testing.assert_array_equal(sim.get_LEFs(), [[0, 21], [79, 99]])
    assert sim.get_statuses()[0, 0] == constants["STATUS_STALLED"]
    assert sim.get_statuses()[1, 1] == constants["STATUS_STALLED"]

    # One motor eventually exhausts the free ring and meets its own other
    # head. Its two stored sites remain distinct; it cannot lap that head.
    sim = ring_at([[0, 1]])
    sim.steps(0, 100)
    left, right = sim.get_LEFs()[0]
    assert (right - left) % 100 == 99
    assert np.all(sim.get_statuses() == constants["STATUS_STALLED"])


@pytest.mark.parametrize("centre", [0, 99])
def test_loading_and_reloading_across_zero(centre):
    n = 100
    loading = np.zeros(n)
    loading[centre] = 1
    # Unload every step and pause every newly loaded head. Each observed
    # state therefore tests the loading geometry without subsequent motion.
    sim = LEFSimulator(1, n, loading, np.ones((n, 5)), np.zeros((n, 2)),
                       np.zeros(n), np.ones(n), circular=True)
    for step in range(30):
        left, right = sim.get_LEFs()[0]
        assert left == (centre - 1) % n
        assert right in (centre, (centre + 1) % n)
        occupied = sim.get_occupied()
        assert occupied[left] == 0
        assert occupied[right] == 1
        assert np.count_nonzero(occupied != constants["OCCUPIED_FREE"]) == 2
        sim.steps(step, step + 1)


def test_directional_capture_and_release_at_zero():
    capture = np.zeros((100, 2))
    capture[0, 1] = 1  # Capture rightward heads; leftward heads remain mobile.
    sim = ring_at([[10, 99]], capture=capture)
    sim.steps(0, 2)
    assert sim.get_LEFs()[0, 1] == 0
    assert sim.get_statuses()[0, 1] == constants["STATUS_CAPTURED"]

    sim = ring_at([[0, 10]], capture=capture)
    sim.steps(0, 1)
    assert sim.get_LEFs()[0, 0] == 99

    release = np.zeros(100)
    release[0] = 1
    sim = ring_at([[10, 0]], capture=capture, release=release)
    sim.steps(0, 1)
    assert sim.get_LEFs()[0, 1] == 1
    assert sim.get_statuses()[0, 1] == constants["STATUS_MOVING"]


def test_ring_graph_distances_include_the_closing_backbone_arc():
    sim = ring_at([[1, 10], [90, 99]])
    sim.populate_dijkstra_arrays(lef_length=0, merge_touching_lefs=False)
    # 10 --cohesin--> 1 --backbone through 0--> 99 --cohesin--> 90.
    assert sim.compute_pair_distances(10, 90) == 2
    assert sim.compute_region_distances(10, 10, 90, 90, True) == (2, 2)
    assert sim.compute_pair_distances(0, 99) == 1
    assert sim.compute_region_distances(98, 2, 45, 46) == 35

    # Neighboring heads across site zero obey the existing optional rule
    # that touching anchors have zero graph separation.
    sim = ring_at([[0, 20], [80, 99]])
    sim.populate_dijkstra_arrays(lef_length=0, merge_touching_lefs=True)
    assert sim.compute_pair_distances(0, 99) == 0


def test_ring_distances_without_cohesin():
    sim = ring_at([])
    sim.populate_dijkstra_arrays(lef_length=0, merge_touching_lefs=False)
    assert sim.compute_pair_distances(0.5, 99.5) == 1
    assert sim.compute_region_distances(98, 2, 4, 6) == 2
    assert sim.compute_region_distances(10, 20, 12, 15) == 0
    assert sim.compute_region_distances(0, 100, 30, 40) == 0
