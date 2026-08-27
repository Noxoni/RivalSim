"""`python -m rivalsim.viewer` entry point."""

try:
    from rivalsim.viewer.app import main
except ModuleNotFoundError as error:
    if error.name == "panda3d" or (error.name and error.name.startswith("direct")):
        raise SystemExit(
            'RivalVis needs the optional viewer dependency. Install it with: '
            'python -m pip install -e ".[viewer]"'
        ) from error
    raise


if __name__ == "__main__":
    raise SystemExit(main())
