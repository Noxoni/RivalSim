# Rival human demonstration recorder plugin

This directory builds the read-only BakkesMod plugin and the native binary-format fixture for
`RIVALRL_NATIVE_DEMO_V1`.

- SDK pin: `sdk.lock.json`
- plugin: `src/rival_demo_recorder.cpp`
- shared C++ format: `include/rivalrec/recording_format.hpp`
- C++/Python fixture: `tests/format_fixture.cpp`
- full installation, commands, schema, fields, limitations, and smoke procedure:
  `../../docs/RIVAL2_HUMAN_DEMO_RECORDER.md`

Configure with `-DBAKKESMOD_SDK_ROOT=<path>` pointing at the exact locked commit. CMake fails if
the checkout does not match.
