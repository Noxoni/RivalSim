"""CPU-only diagnostic of frozen checkpoints on frozen native scenario starts.

No physics rollout, optimizer, training mutation, new reward, or skill detector.
CPU recomputation uses the production observation builder and actor; it is not
claimed to be bit-identical to a captured GPU forward pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from benchmarks.run_rival2_fresh_ground_30hz_v1 import RESULTS, CHECKPOINTS, sha, write_json, utc
from rivalsim.fresh_ground_30hz import SEED, scenarios, scenario_hash, CHECKPOINT_FORMAT
from rivalsim.rival2_env import Rival2WorldSim, Rival2TensorBridge
from rivalsim.rival2_independent_critic import IndependentCriticActorCritic, IndependentCriticPolicyConfig
from rivalsim.rival2_unified_policy import deterministic_unified_action


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", nargs="+", type=int, default=[0, 50, 100, 150])
    args = parser.parse_args()
    torch.set_num_threads(2)
    corpus = {}
    for index, (name, family) in enumerate((("acquisition_selfplay", 2), ("finishing_selfplay", 4),
                                           ("standard_kickoff_nexto", 1))):
        bank = scenarios(64, SEED+100+index, family_only=family)
        world = Rival2WorldSim(64, "G:/dev/RLBot-Rival/bot/collision_meshes", device="cpu",
            reward_mode=2, physics_ticks_per_decision=4, ssl_foundation_scenarios=bank)
        bridge = Rival2TensorBridge(world)
        obs = bridge.observation()
        rows, sides = torch.arange(64), torch.as_tensor(bank.focal_side.astype("int64"))
        delta = torch.as_tensor(bank.state.ball_pos[:, None]-bank.state.car_pos)[rows, sides]
        forward, _ = bridge._basis(torch.as_tensor(bank.state.car_quat)[rows, sides])
        nose_ball_cos = (forward*delta).sum(-1)/delta.norm(dim=-1)
        corpus[name] = dict(observation=obs, scenario_sha256=scenario_hash(bank), focal_side=sides,
                            initial_ball_distance=delta.norm(dim=-1), initial_nose_ball_cosine=nose_ball_cos,
                            kickoff_layout=bank.kickoff_layout.tolist())
    report = {"utc": utc(), "device": "cpu", "optimizer_steps": 0, "physics_ticks": 0,
              "method": "production native initial-state observation builder; zero initial GRU; CPU actor recomputation, not captured GPU bitwise parity",
              "scenarios": {name: {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in case.items()}
                            for name, case in corpus.items()}, "checkpoints": {}}
    for update in args.updates:
        path = CHECKPOINTS / ("initial.pt" if update == 0 else f"u{update:06d}.pt")
        before_hash = sha(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["format"] == CHECKPOINT_FORMAT and payload["accepted_updates_total"] == update
        model = IndependentCriticActorCritic(IndependentCriticPolicyConfig(**payload["policy_config"]))
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        entry = {"path": str(path), "sha256": before_hash, "cases": {}}
        for name, case in corpus.items():
            obs, side = case["observation"], case["focal_side"]
            actor, _ = model.forward_actor(obs.reshape(-1, 182), model.initial_hidden(128, device="cpu"),
                                           reset_before=torch.ones(128, dtype=torch.bool))
            actor = actor.reshape(64, 2, 13)[torch.arange(64), side]
            action = deterministic_unified_action(actor)
            assert bool(torch.isfinite(actor).all())
            probabilities = actor[:, 10:13].sigmoid()
            entry["cases"][name] = {
                "mean_deterministic_action": action.mean(0).tolist(),
                "minimum_throttle": float(action[:, 0].min()), "maximum_throttle": float(action[:, 0].max()),
                "throttle_gt_point5_fraction": float((action[:, 0] > .5).float().mean()),
                "throttle_negative_fraction": float((action[:, 0] < 0).float().mean()),
                "boost_on_fraction": float(action[:, 6].mean()), "jump_on_fraction": float(action[:, 5].mean()),
                "handbrake_on_fraction": float(action[:, 7].mean()),
                "mean_button_probabilities": probabilities.mean(0).tolist(),
                "minimum_button_logit_abs": float(actor[:, 10:13].abs().min()),
                "per_state_action": action.tolist(), "per_state_button_probability": probabilities.tolist(),
                "per_state_analog_std": actor[:, 5:10].clamp(model.config.log_std_min, model.config.log_std_max).exp().tolist(),
            }
        assert sha(path) == before_hash
        report["checkpoints"][str(update)] = entry
        del model, payload
    report["torch_cuda_bytes_allocated"] = torch.cuda.memory_allocated()
    assert report["torch_cuda_bytes_allocated"] == 0
    out = RESULTS / "monitoring" / f"initial_action_diagnostic_through_u{max(args.updates):06d}.json"
    write_json(out, report)
    for update, entry in report["checkpoints"].items():
        for name, case in entry["cases"].items():
            print(update, name, {k: v for k, v in case.items() if not k.startswith("per_state")})


if __name__ == "__main__":
    main()
