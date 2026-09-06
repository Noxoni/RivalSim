"""Native bridge interface/cadence checks; not evidence of actual gameplay."""
import pytest
import torch

from rivalsim.entity_native_bridge import EntityNativeActor, RecurrentPacketScheduler
from rivalsim.ssl_entity_policy import EntityJointControlActorCritic


class CounterActor:
    def __call__(self, obs, hidden):
        return torch.zeros(1, 8), hidden + 1, torch.zeros(1, 90)


def feed(scheduler, frame, *, identity="a", active=True, events=None):
    return scheduler.step(frame=frame, identity=identity, active=active,
                          observation=lambda: torch.zeros(1, 182),
                          lifecycle=lambda delta, reset: events.append((frame, delta, reset)) if events is not None else None)


def test_four_tick_hold_lifecycle_and_recurrent_counts():
    s = RecurrentPacketScheduler(CounterActor(), 2)
    events = []
    for frame in range(120):
        feed(s, frame, events=events)
    assert len(events) == 120 and events[0] == (0, 0, True)
    assert s.stats["decisions"] == 30 and s.stats["held"] == 90
    assert torch.equal(s.hidden, torch.full((1, 1, 2), 30.))


def test_duplicates_and_out_of_order_do_not_advance_lifecycle_or_hidden():
    s = RecurrentPacketScheduler(CounterActor(), 2)
    events = []
    for frame in (10, 10, 9, 11, 14):
        feed(s, frame, events=events)
    assert len(events) == 3 and s.stats["decisions"] == 2
    assert s.stats["duplicates"] == s.stats["out_of_order"] == 1


def test_gap_is_reported_without_fabricating_hidden_or_decisions():
    s = RecurrentPacketScheduler(CounterActor(), 2)
    for frame in (0, 1, 17, 18, 21):
        feed(s, frame)
    assert s.stats["missed_ticks"] == 17
    assert s.stats["skipped_decisions"] == 3
    assert s.stats["decisions"] == 3
    assert float(s.hidden.max()) == 3


@pytest.mark.parametrize("reset_kind", ["identity", "pause"])
def test_reset_clears_memory_and_held_controls(reset_kind):
    s = RecurrentPacketScheduler(CounterActor(), 2)
    feed(s, 100)
    feed(s, 104)
    if reset_kind == "pause":
        assert not feed(s, 105, active=False).any()
        feed(s, 106)
    else:
        feed(s, 0, identity="newmatch")
    assert s.stats["resets"] == 2 and float(s.hidden.max()) == 1


def test_nonfinite_output_does_not_commit_recurrent_state():
    s = RecurrentPacketScheduler(CounterActor(), 2)
    feed(s, 0)
    prior = s.hidden.clone()
    s.model = lambda o, h: (torch.full((1, 8), float("nan")), h+100, torch.zeros(1, 90))
    with pytest.raises(RuntimeError, match="Invalid actor output"):
        feed(s, 4)
    assert torch.equal(s.hidden, prior) and s.stats["decisions"] == 1


def test_observation_not_built_on_held_frames():
    s = RecurrentPacketScheduler(CounterActor(), 2)
    feed(s, 0)
    def forbidden():
        raise AssertionError("held frame must not request policy observation")
    s.step(frame=1, identity="a", active=True, observation=forbidden, lifecycle=lambda d,r: None)


def test_scripted_actor_matches_eager_and_has_no_critic_or_optimizer():
    torch.set_num_threads(2)
    torch.manual_seed(987)
    source = EntityJointControlActorCritic().eval()
    wrapper = EntityNativeActor(source).eval()
    script = torch.jit.script(wrapper)
    for batch in (1, 3):
        hidden = source.initial_hidden(batch, device="cpu")
        with torch.inference_mode():
            for _ in range(6):
                obs = torch.randn(batch, 182)
                expected, eh = source.forward_actor(obs, hidden)
                action, next_hidden, logits = script(obs, hidden)
                torch.testing.assert_close(logits, expected, atol=2e-5, rtol=1e-6)
                torch.testing.assert_close(next_hidden, eh, atol=2e-6, rtol=1e-6)
                assert torch.equal(action, source.deterministic(expected))
                hidden = eh
    assert not any("critic" in k for k in script.state_dict())
    assert all(not p.requires_grad for p in wrapper.parameters())
    assert source.actor.weight.requires_grad


def test_script_rejects_wrong_observation_width():
    script = torch.jit.script(EntityNativeActor(EntityJointControlActorCritic().eval()))
    with pytest.raises(torch.jit.Error):
        script(torch.zeros(1, 181), torch.zeros(1, 1, script.hidden_dim))


def test_field_availability_complete_and_never_promotes_missing_wheels():
    from benchmarks.audit_entity_native_inputs import classify
    from rivalsim.rival2_contracts import OBS_FIELD_NAMES
    rows = [classify(name) for name in OBS_FIELD_NAMES]
    assert len(rows) == 182 and all(note for _,note in rows)
    missing = [name for name in OBS_FIELD_NAMES if classify(name)[0] == "unavailable"]
    assert len(missing) == 10
    assert all("wheel_contact" in name or "sticky_ticks" in name for name in missing)
    with pytest.raises(ValueError):
        classify("self.new_unclassified_field")
