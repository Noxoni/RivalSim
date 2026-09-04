from __future__ import annotations

from benchmarks import run_rival2_ssl_foundation_ppo_v2 as campaign


def test_corrected_campaign_has_isolated_fresh_launch_namespace() -> None:
    assert campaign.FORMAT == "RIVAL2_SSL_FOUNDATION_PPO_V2"
    assert campaign.RESULTS.name == "ssl_foundation_ppo_v2"
    assert campaign.AUTHORITY.name == "authority.json"
    assert campaign.SCHEDULE_AUTHORITY.name == "launch_authority.json"
    assert campaign.CHECKPOINT.parent.name == "ssl_foundation_ppo_v2"
    assert campaign.DEFAULT_RUN_DIR.name == "ssl-foundation-ppo-v2"
    assert campaign.SNAPSHOT_INTERVAL == 50
    assert campaign.CONTINUATION_REVIEW_MARKER == 600


def test_corrected_authority_is_v5_rooted_and_has_no_resume_parent() -> None:
    payload = campaign.authority_payload("A" * 40)
    transition = payload["source_transition"]
    assert payload["source"]["sha256"] == (
        "955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216"
    )
    assert transition["initial_accepted_ppo_updates"] == 0
    assert transition["initial_resume_checkpoint"] is None
    assert transition["fresh_optimizer"] is True
    assert transition["forbidden_lineage"] == "RIVAL2_SSL_FOUNDATION_PPO_V1"
    assert payload["reset_curriculum"]["heading_generation"] == {
        "global_face_ball_postprocess": False,
        "ground_heading_momentum": "coherent_with_off_angle_coverage",
        "intentionally_aligned_families": ["shooting_finishing", "contested_fifty"],
    }
    variants = payload["reset_curriculum"]["wall_aerial_variants"]
    assert set(variants) == {
        "grounded_elevated_ball",
        "side_wall_car",
        "airborne_intercept",
    }
    assert all(count > 0 for count in variants.values())
    assert payload["campaign"]["snapshot_interval"] == 50
    assert payload["campaign"]["evaluation_interval"] == 50


def test_v2_engine_configuration_removes_update_20_binding() -> None:
    campaign._configure_engine()
    assert campaign.engine.BOUND_RESUME_UPDATE is None
    assert campaign.engine.BOUND_RESUME_SHA256 is None
    assert campaign.engine.AUTHORITY == campaign.AUTHORITY
    assert campaign.engine.SCHEDULE_AUTHORITY == campaign.SCHEDULE_AUTHORITY
    assert campaign.engine.SOURCE_SHA256 == (
        "955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216"
    )
