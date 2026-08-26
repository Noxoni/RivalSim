"""RivalSim public API."""

from rivalsim.arena import ArenaGeometry
from rivalsim.controls import ControlBatch
from rivalsim.rival2_contracts import OBS_DIM
from rivalsim.rival2_env import Rival2Env, Rival2Step, Rival2TensorBridge, Rival2WorldSim
from rivalsim.rival2_full_match_env import (
    Rival2FullMatchEnv,
    Rival2FullMatchState,
    Rival2FullMatchWorldSim,
)
from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig
from rivalsim.rival2_ppo import Rival2PPOConfig, Rival2RolloutBuffer
from rivalsim.rival2_training import Rival2SelfPlayConfig, Rival2Trainer
from rivalsim.simulator import RivalSim
from rivalsim.state import StateSnapshot
from rivalsim.static_world import (
    ActionTape,
    CarCarWorldSim,
    CompleteWorldSim,
    DynamicWorldSim,
    IntegratedWorldSim,
    StaticWorldSim,
    make_standard_kickoff_state,
)

__all__ = [
    "OBS_DIM",
    "ActionTape",
    "ArenaGeometry",
    "CarCarWorldSim",
    "CompleteWorldSim",
    "ControlBatch",
    "DynamicWorldSim",
    "IntegratedWorldSim",
    "Rival2ActorCritic",
    "Rival2Env",
    "Rival2FullMatchEnv",
    "Rival2FullMatchState",
    "Rival2FullMatchWorldSim",
    "Rival2PPOConfig",
    "Rival2PolicyConfig",
    "Rival2RolloutBuffer",
    "Rival2SelfPlayConfig",
    "Rival2Step",
    "Rival2TensorBridge",
    "Rival2Trainer",
    "Rival2WorldSim",
    "RivalSim",
    "StateSnapshot",
    "StaticWorldSim",
    "make_standard_kickoff_state",
]
__version__ = "0.5.0"
