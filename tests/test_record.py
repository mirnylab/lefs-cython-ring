"""The sparse 3D spring history must preserve the original motor dynamics."""
import numpy as np
import pytest

from lefs_cython.simple import LEFSimulator, constants, seed_random


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("move_policy", [0, 1])
def test_record_replays_every_tick_without_changing_random_draws(circular, move_policy):
    # A crowded, short backbone exercises loading, collisions, capture/release,
    # and pausing. These are verification settings, not biological parameters.
    n, n_motors, n_steps = 128, 12, 600
    capture = np.zeros((n, 2))
    capture[::8, 0] = 0.7
    capture[4::8, 1] = 0.7
    unload = np.full((n, 5), 0.03)
    unload[:, constants["STATUS_CAPTURED"]] = 0.01

    def make_simulator():
        seed_random(42)
        return LEFSimulator(
            n_motors, n, np.ones(n), unload, capture, np.full(n, 0.05),
            np.full(n, 0.35), circular=circular, move_policy=move_policy,
            alternate_prob=0.2,
        )

    reference = make_simulator()
    initial = reference.get_LEFs()
    expected_positions, expected_statuses = [], []
    for tick in range(n_steps):
        reference.steps(tick, tick + 1)
        expected_positions.append(reference.get_LEFs())
        expected_statuses.append(reference.get_statuses())
    expected_positions = np.asarray(expected_positions)
    expected_statuses = np.asarray(expected_statuses)
    expected_occupied = reference.get_occupied()

    # Continuing after the comparison also checks that recording has not
    # changed the RNG state, rather than merely reaching matching endpoints.
    reference.steps(n_steps, n_steps + 37)
    continued = (reference.get_LEFs(), reference.get_statuses(), reference.get_occupied())

    recorded = make_simulator()
    np.testing.assert_array_equal(recorded.get_LEFs(), initial)
    chunks = []
    for start, end in [(0, 1), (1, 73), (73, n_steps)]:
        chunks.append(recorded.steps_record(start, end))
        np.testing.assert_array_equal(recorded.get_statuses(), expected_statuses[end - 1])
    records = np.concatenate(chunks)
    assert records.dtype == np.int64
    np.testing.assert_array_equal(recorded.get_occupied(), expected_occupied)

    # Replay the changed spring endpoints, checking every intermediate tick.
    # Each slot appears at most once per tick, even if it unloaded and reloaded.
    positions = initial.copy()
    row = 0
    for tick in range(n_steps):
        previous_slot = -1
        while row < len(records) and records[row, 0] == tick:
            _, slot, left, right = records[row]
            assert slot > previous_slot
            assert np.any(positions[slot] != [left, right])
            positions[slot] = left, right
            previous_slot = slot
            row += 1
        np.testing.assert_array_equal(positions, expected_positions[tick])
    assert row == len(records)

    # Ensure this deterministic fixture really exercised the relevant events.
    assert constants["STATUS_STALLED"] in expected_statuses
    assert constants["STATUS_CAPTURED"] in expected_statuses
    changes = np.diff(np.concatenate([initial[None], expected_positions]), axis=0)
    if circular:
        # Ordinary outward movement is 0 or -1 modulo N for leg 0. A different
        # change demonstrates immediate unloading/reloading in a fixed slot.
        assert np.any(~np.isin(changes[:, :, 0] % n, [0, n - 1]))
    else:
        assert np.any(np.abs(changes) > 1)

    recorded.steps(n_steps, n_steps + 37)
    for actual, expected in zip(
        (recorded.get_LEFs(), recorded.get_statuses(), recorded.get_occupied()), continued
    ):
        np.testing.assert_array_equal(actual, expected)


def test_record_keeps_ring_leg_identity_int64_times_and_omits_status_only_changes():
    n = 100
    capture = np.zeros((n, 2))
    capture[40, 0] = capture[50, 1] = 1
    sim = LEFSimulator(
        2, n, np.ones(n), np.zeros((n, 5)), capture, np.zeros(n),
        np.zeros(n), circular=True, skip_load=True,
    )
    sim.force_load_LEFs(np.array([[0, 10], [40, 50]], dtype=np.int32))
    tick = 2**31 + 17  # The history clock must not wrap at the int32 limit.
    records = sim.steps_record(tick, tick + 1)
    np.testing.assert_array_equal(records, [[tick, 0, 99, 11]])
    np.testing.assert_array_equal(sim.get_LEFs(), [[99, 11], [40, 50]])
    assert np.all(sim.get_statuses()[1] == constants["STATUS_CAPTURED"])

    positions, statuses = sim.get_LEFs(), sim.get_statuses()
    assert sim.steps_record(tick + 1, tick + 1).shape == (0, 4)
    np.testing.assert_array_equal(sim.get_LEFs(), positions)
    np.testing.assert_array_equal(sim.get_statuses(), statuses)

    # Capture both remaining heads: no endpoint changes, despite status changes.
    all_captured = LEFSimulator(
        1, n, np.ones(n), np.zeros((n, 5)), np.ones((n, 2)),
        np.zeros(n), np.zeros(n), circular=True, skip_load=True,
    )
    all_captured.force_load_LEFs(np.array([[10, 20]], dtype=np.int32))
    empty = all_captured.steps_record(0, 20)
    assert empty.shape == (0, 4)
    assert empty.dtype == np.int64
    assert np.all(all_captured.get_statuses() == constants["STATUS_CAPTURED"])
