"""Build the frozen RivalSim v0.4 native lifecycle authority cache."""

from __future__ import annotations

import argparse
import json

from rivalsim.v04_authority import authority_identity, build_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("collision_root")
    parser.add_argument("--cache-root", default=".tools/v0.4/oracle")
    args = parser.parse_args()
    directory, authority = build_cache(args.collision_root, args.cache_root)
    identity, _inputs = authority_identity(args.collision_root)
    print(
        json.dumps(
            {
                "identity": identity,
                "directory": str(directory),
                "pad_cases": len(authority["boost_pads"]["pickup_cases"]),
                "goal_cases": len(authority["goals_kickoff"]["boundary_cases"]),
                "kickoff_cases": len(authority["goals_kickoff"]["kickoff_cases"]),
                "respawn_poses": len(authority["demolition_respawn"]["respawn_poses"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
