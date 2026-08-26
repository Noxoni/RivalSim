# Active Codex Handoff — Rival 2.0 Campaign 04

Campaign 03 is complete. Campaign 04 is authorized as a direct long-run continuation of the exact Campaign 03 100M checkpoint.

Start from current `origin/main` and read `handoff/rival2-c04/README.md` in full. Treat it as the controlling requirement.

Required completed Campaign 03 parent:

`67b51452df98696a54f4465ea83924c6b9e75b4d`

Resume:

`checkpoints/rival2/campaign03/rival2_campaign03_100m_resume.pt`

Expected checkpoint SHA-256:

`A0F2E554448B31A373BD73254125AC0ADFDB541EE3B695AD9D040B2CCFA68991`

Mission:

1. load that checkpoint with its optimizer/RNG/counters/historical state intact;
2. immediately continue the unchanged Reward V2 / entropy-off training line at 131,072 worlds and horizon 32;
3. train through update 120 / 1,006,632,960 cumulative agent decision samples, the first completed update crossing 1B;
4. save and evaluate checkpoints at updates 30, 60, 90, and 120 (250M/500M/750M/1B crossings) using the same 4,096-world stochastic self-play protocol from Campaign 03;
5. publish a compact behavioral curve against the existing Campaign 03 100M baseline and the exact final resumable checkpoint;
6. stop.

Do not run a preflight, reward smoke, initialization evaluation, world-count sweep, inherited parity/regression suite, post-run pytest/Ruff/compileall ceremony, or extra evaluation outside the four authorized checkpoints. Do not change any reward, PPO, model, observation, action, episode, self-play, or simulator setting.

Do not build the viewer yet. Viewer work is deferred until after the 1B result is reviewed. Do not begin v0.6.