"""GPU-resident v0.3 two-Octane pair state and event diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

MAX_CAR_CAR_CONTACTS = 4
MAX_CAR_BUMP_EVENTS_PER_TICK = 4
CAR_VISIT_A_THEN_B = 0
CAR_VISIT_B_THEN_A = 1


def _normalize_visit_order(
    value: str | int | np.ndarray,
    num_envs: int,
) -> np.ndarray:
    if isinstance(value, str):
        names = {
            "a_then_b": CAR_VISIT_A_THEN_B,
            "b_then_a": CAR_VISIT_B_THEN_A,
        }
        try:
            value = names[value]
        except KeyError as exc:
            raise ValueError(
                "car visitation order must be 'a_then_b' or 'b_then_a'"
            ) from exc
    result = np.broadcast_to(np.asarray(value, dtype=np.int32), (num_envs,)).copy()
    if np.any((result != CAR_VISIT_A_THEN_B) & (result != CAR_VISIT_B_THEN_A)):
        raise ValueError("car visitation order entries must be 0 (A->B) or 1 (B->A)")
    return result


def _lifecycle_orders(seed: int, epochs: np.ndarray) -> np.ndarray:
    """Choose a generic order from per-world lifecycle state, not physics data."""

    mask = (1 << 64) - 1
    result = np.empty(len(epochs), dtype=np.int32)
    for world, epoch in enumerate(np.asarray(epochs, dtype=np.uint64)):
        # SplitMix64 is used only as a deterministic internal lifecycle source.
        # It deliberately has no access to case IDs, physical state, or expected
        # results. A membership event advances ``epoch`` before this is called.
        x = (
            (int(seed) & mask)
            + 0x9E3779B97F4A7C15 * (world + 1)
            + 0xD1B54A32D192ED03 * int(epoch)
        ) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        result[world] = (x ^ (x >> 31)) & 1
    return result


@dataclass(slots=True)
class CarCarSnapshot:
    pre_tick_first_car: np.ndarray
    membership_epoch: np.ndarray
    contact_count: np.ndarray
    return_code: np.ndarray
    contact_point_b_bt: np.ndarray
    contact_normal: np.ndarray
    contact_distance_bt: np.ndarray
    event_count: np.ndarray
    event_bumper: np.ndarray
    event_victim: np.ndarray
    event_is_demo: np.ndarray
    car_contact_id: np.ndarray
    car_contact_cooldown: np.ndarray
    car_is_demoed: np.ndarray


class CarCarState:
    """Persistent semantic state for one fixed Octane pair per world."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        *,
        lifecycle_seed: int = 0,
        pre_tick_first_car: str | int | np.ndarray | None = None,
        membership_epoch: np.ndarray | None = None,
    ):
        self.num_envs = num_envs
        self.device = device
        self.lifecycle_seed = int(lifecycle_seed)
        if membership_epoch is None:
            # Construction of the fixed pair is the first membership epoch.
            self._membership_epoch = np.ones(num_envs, dtype=np.uint64)
        else:
            self._membership_epoch = np.asarray(
                membership_epoch, dtype=np.uint64
            ).reshape(num_envs).copy()
        if pre_tick_first_car is None:
            self._pre_tick_first_car = _lifecycle_orders(
                self.lifecycle_seed, self._membership_epoch
            )
        else:
            self._pre_tick_first_car = _normalize_visit_order(
                pre_tick_first_car, num_envs
            )
        self.pre_tick_first_car = wp.array(
            self._pre_tick_first_car, dtype=wp.int32, device=device
        )
        for name in ("contact_count", "return_code", "algorithm_active", "event_count"):
            setattr(self, name, wp.zeros(num_envs, dtype=wp.int32, device=device))
        car_capacity = num_envs * 2
        for name in (
            "pre_position_bt",
            "pre_velocity_bt",
            "pre_angular_velocity",
            "queued_velocity_bt",
        ):
            setattr(self, name, wp.zeros(car_capacity, dtype=wp.vec3, device=device))
        self.pre_quaternion = wp.zeros(car_capacity, dtype=wp.quat, device=device)
        self.pre_on_ground = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        self.pre_is_supersonic = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        self.pre_supersonic_time = wp.zeros(
            car_capacity, dtype=wp.float32, device=device
        )
        self.car_contact_id = wp.full(car_capacity, -1, dtype=wp.int32, device=device)
        self.car_contact_cooldown = wp.zeros(
            car_capacity, dtype=wp.float32, device=device
        )
        self.car_is_demoed = wp.zeros(car_capacity, dtype=wp.int32, device=device)
        contact_capacity = num_envs * MAX_CAR_CAR_CONTACTS
        for name in (
            "manifold_local_a_bt",
            "manifold_local_b_bt",
            "manifold_normal",
            "manifold_tangent",
            "contact_point_b_bt",
            "contact_normal",
        ):
            setattr(self, name, wp.zeros(contact_capacity, dtype=wp.vec3, device=device))
        for name in (
            "manifold_distance_bt",
            "manifold_normal_jacobian",
            "manifold_tangent_jacobian",
            "manifold_normal_rhs",
            "manifold_tangent_rhs",
            "manifold_push_rhs",
            "manifold_normal_impulse",
            "manifold_tangent_impulse",
            "manifold_push_impulse",
            "contact_distance_bt",
        ):
            setattr(
                self,
                name,
                wp.zeros(contact_capacity, dtype=wp.float32, device=device),
            )
        event_capacity = num_envs * MAX_CAR_BUMP_EVENTS_PER_TICK
        self.event_bumper = wp.full(
            event_capacity, -1, dtype=wp.int32, device=device
        )
        self.event_victim = wp.full(
            event_capacity, -1, dtype=wp.int32, device=device
        )
        self.event_is_demo = wp.zeros(
            event_capacity, dtype=wp.int32, device=device
        )

    @property
    def logical_bytes(self) -> int:
        car = 2 * (4 * 3 + 4 + 4) * 4
        contact = MAX_CAR_CAR_CONTACTS * (6 * 3 + 11) * 4
        events = MAX_CAR_BUMP_EVENTS_PER_TICK * 3 * 4
        return self.num_envs * (5 * 4 + car + contact + events)

    @property
    def membership_epoch(self) -> np.ndarray:
        return self._membership_epoch.copy()

    @property
    def visit_order(self) -> np.ndarray:
        return self._pre_tick_first_car.copy()

    def membership_changed(
        self,
        pre_tick_first_car: str | int | np.ndarray | None = None,
    ) -> None:
        """Establish the next order after a real container-membership change."""

        self._membership_epoch += np.uint64(1)
        if pre_tick_first_car is None:
            self._pre_tick_first_car = _lifecycle_orders(
                self.lifecycle_seed, self._membership_epoch
            )
        else:
            self._pre_tick_first_car = _normalize_visit_order(
                pre_tick_first_car, self.num_envs
            )
        self.pre_tick_first_car = wp.array(
            self._pre_tick_first_car, dtype=wp.int32, device=self.device
        )

    def lifecycle_copy_kwargs(self) -> dict[str, object]:
        """Return only the state that survives a non-membership world reset."""

        return {
            "lifecycle_seed": self.lifecycle_seed,
            "pre_tick_first_car": self._pre_tick_first_car,
            "membership_epoch": self._membership_epoch,
        }

    def snapshot(self) -> CarCarSnapshot:
        count = self.num_envs

        def array(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
            return np.asarray(getattr(self, name).numpy(), dtype=dtype).reshape(shape)

        return CarCarSnapshot(
            pre_tick_first_car=self.visit_order,
            membership_epoch=self.membership_epoch,
            contact_count=array("contact_count", np.int32, (count,)),
            return_code=array("return_code", np.int32, (count,)),
            contact_point_b_bt=array(
                "contact_point_b_bt",
                np.float32,
                (count, MAX_CAR_CAR_CONTACTS, 3),
            ),
            contact_normal=array(
                "contact_normal", np.float32, (count, MAX_CAR_CAR_CONTACTS, 3)
            ),
            contact_distance_bt=array(
                "contact_distance_bt",
                np.float32,
                (count, MAX_CAR_CAR_CONTACTS),
            ),
            event_count=array("event_count", np.int32, (count,)),
            event_bumper=array(
                "event_bumper", np.int32, (count, MAX_CAR_BUMP_EVENTS_PER_TICK)
            ),
            event_victim=array(
                "event_victim", np.int32, (count, MAX_CAR_BUMP_EVENTS_PER_TICK)
            ),
            event_is_demo=array(
                "event_is_demo", np.int32, (count, MAX_CAR_BUMP_EVENTS_PER_TICK)
            ),
            car_contact_id=array("car_contact_id", np.int32, (count, 2)),
            car_contact_cooldown=array(
                "car_contact_cooldown", np.float32, (count, 2)
            ),
            car_is_demoed=array("car_is_demoed", np.int32, (count, 2)),
        )


__all__ = [
    "CAR_VISIT_A_THEN_B",
    "CAR_VISIT_B_THEN_A",
    "MAX_CAR_BUMP_EVENTS_PER_TICK",
    "MAX_CAR_CAR_CONTACTS",
    "CarCarSnapshot",
    "CarCarState",
]
