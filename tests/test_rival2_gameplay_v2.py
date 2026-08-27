from __future__ import annotations

import numpy as np
import warp as wp

from rivalsim.kernels.rival2 import (
    rival2_reset_strict_dash_state,
    rival2_track_strict_double_dash,
)
from rivalsim.rival2_contracts import (
    GAMEPLAY_STRICT_DOUBLE_DASH_REWARD,
    REWARD_GAMEPLAY_V1_CONTRACT_HASH,
    REWARD_GAMEPLAY_V2_CONTRACT,
    RIVAL2_REWARD_GAMEPLAY_V2_VERSION,
    contract_hashes_for_reward,
)


class _StrictDashHarness:
    def __init__(self) -> None:
        self.device = "cpu"
        self.episode_ticks = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.on_ground = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.has_flipped = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.air_time = wp.zeros(2, dtype=wp.float32, device=self.device)
        self.wheel_contact = wp.zeros(8, dtype=wp.int32, device=self.device)
        self.previous_on_ground = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.previous_has_flipped = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.previous_air_time = wp.zeros(2, dtype=wp.float32, device=self.device)
        self.previous_wheel_mask = wp.zeros(2, dtype=wp.int32, device=self.device)
        self.pending = wp.full(2, -1, dtype=wp.int32, device=self.device)
        self.last_success_flip = wp.full(2, -1, dtype=wp.int32, device=self.device)
        self.last_success_landing = wp.full(2, -1, dtype=wp.int32, device=self.device)
        self.count = wp.zeros(2, dtype=wp.int32, device=self.device)
        reset = wp.ones(1, dtype=wp.int32, device=self.device)
        wp.launch(
            rival2_reset_strict_dash_state,
            dim=2,
            inputs=[
                reset,
                self.on_ground,
                self.has_flipped,
                self.air_time,
                self.wheel_contact,
                self.previous_on_ground,
                self.previous_has_flipped,
                self.previous_air_time,
                self.previous_wheel_mask,
                self.pending,
                self.last_success_flip,
                self.last_success_landing,
            ],
            device=self.device,
        )

    def step(
        self,
        tick: int,
        *,
        flipped: int,
        wheel_mask: int,
        on_ground: int = 0,
        air_time: float = 0.0,
    ) -> None:
        self.episode_ticks.assign(np.asarray((tick - 1,), dtype=np.int32))
        flipped_values = np.zeros(2, dtype=np.int32)
        flipped_values[0] = flipped
        self.has_flipped.assign(flipped_values)
        ground_values = np.zeros(2, dtype=np.int32)
        ground_values[0] = on_ground
        self.on_ground.assign(ground_values)
        air_values = np.zeros(2, dtype=np.float32)
        air_values[0] = air_time
        self.air_time.assign(air_values)
        wheels = np.zeros(8, dtype=np.int32)
        for wheel in range(4):
            wheels[wheel] = int((wheel_mask & (1 << wheel)) != 0)
        self.wheel_contact.assign(wheels)
        wp.launch(
            rival2_track_strict_double_dash,
            dim=2,
            inputs=[
                self.episode_ticks,
                self.on_ground,
                self.has_flipped,
                self.air_time,
                self.wheel_contact,
                self.previous_on_ground,
                self.previous_has_flipped,
                self.previous_air_time,
                self.previous_wheel_mask,
                self.pending,
                self.last_success_flip,
                self.last_success_landing,
                self.count,
            ],
            device=self.device,
        )

    def value(self) -> int:
        return int(np.asarray(self.count.numpy())[0])


def _first_retained_real_pair(harness: _StrictDashHarness, second_tick: int = 737) -> None:
    # Gameplay V1 +180 stochastic trace, world 206, Rival Blue:
    # flip 673 -> landing 675, then flip 737 -> landing 744.
    harness.step(672, flipped=0, wheel_mask=0, air_time=2 / 120)
    harness.step(673, flipped=1, wheel_mask=0, air_time=2 / 120)
    harness.step(675, flipped=1, wheel_mask=0b1110, air_time=0.0)
    harness.step(second_tick - 1, flipped=0, wheel_mask=0, air_time=15 / 120)
    harness.step(second_tick, flipped=1, wheel_mask=0, air_time=15 / 120)
    harness.step(second_tick + 7, flipped=1, wheel_mask=0b0001, air_time=0.0)


def test_gameplay_v2_contract_is_v1_plus_only_strict_double_dash() -> None:
    assert GAMEPLAY_STRICT_DOUBLE_DASH_REWARD == 0.005
    assert REWARD_GAMEPLAY_V2_CONTRACT["base_reward"]["version"] == "RIVAL2_REWARD_GAMEPLAY_V1"
    assert REWARD_GAMEPLAY_V1_CONTRACT_HASH == (
        "48AAC000B97D2652507F677184A3FE4F0A3A86CED136B680C933EFF33CD9F072"
    )
    assert RIVAL2_REWARD_GAMEPLAY_V2_VERSION in contract_hashes_for_reward(
        RIVAL2_REWARD_GAMEPLAY_V2_VERSION
    )


def test_online_strict_dash_matches_retained_real_rival_pair() -> None:
    harness = _StrictDashHarness()
    _first_retained_real_pair(harness)
    assert harness.value() == 1


def test_online_strict_dash_rejects_single_wavedash_and_long_pair() -> None:
    single = _StrictDashHarness()
    single.step(672, flipped=0, wheel_mask=0, air_time=2 / 120)
    single.step(673, flipped=1, wheel_mask=0, air_time=2 / 120)
    single.step(675, flipped=1, wheel_mask=0b1110)
    assert single.value() == 0

    long_pair = _StrictDashHarness()
    _first_retained_real_pair(long_pair, second_tick=764)
    assert long_pair.value() == 0


def test_online_strict_dash_rejects_high_air_and_late_landing() -> None:
    high_air = _StrictDashHarness()
    high_air.step(99, flipped=0, wheel_mask=0, air_time=43 / 120)
    high_air.step(100, flipped=1, wheel_mask=0, air_time=43 / 120)
    high_air.step(102, flipped=1, wheel_mask=0b1111)
    high_air.step(110, flipped=0, wheel_mask=0, air_time=0.1)
    high_air.step(111, flipped=1, wheel_mask=0, air_time=0.1)
    high_air.step(113, flipped=1, wheel_mask=0b1111)
    assert high_air.value() == 0

    late = _StrictDashHarness()
    late.step(99, flipped=0, wheel_mask=0, air_time=0.1)
    late.step(100, flipped=1, wheel_mask=0, air_time=0.1)
    late.step(125, flipped=1, wheel_mask=0b1111)
    late.step(130, flipped=0, wheel_mask=0, air_time=0.1)
    late.step(131, flipped=1, wheel_mask=0, air_time=0.1)
    late.step(133, flipped=1, wheel_mask=0b1111)
    assert late.value() == 0
