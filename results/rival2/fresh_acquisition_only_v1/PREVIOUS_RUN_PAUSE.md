# Previous mixed-scenario campaign preserved, not used as parent

At the user's request for an isolated fresh-model experiment, a STOP marker was
written to `G:/dev/RivalSim-runs/sustained-acquisition-v1/STOP`. The active update
finished and the runner exited normally at accepted update 156. Both worker and
launcher were then absent. The old monitor was paused through the Codex automation
API. No old STOP marker was cleared.

The exact latest rolling file was copied without reserialization to:
`checkpoints/rival2/sustained_acquisition_v1/plus_000156.pt`.
SHA-256: `0488EE74260FF5887075D052B5478E075A55D192203B3A9D3879CD56BCBE5818`.
It retains 690,094,080 learner decisions and 22,528 optimizer steps. The existing
CPU audit passed; this model and optimizer do not initialize the new experiment.

The last completed full-match evaluation was update 150:
checkpoint SHA `F62C2AB0257482DC8E5643D432269DEA4B746448BB631482FBD03E511007E89A`.
Ten complete Nexto matches: zero wins, zero goals scored, 364 conceded, 179 Rival
contacts (3.58/minute), no entirely touchless matches, 89 kickoff first contacts.
One of 175 resolved followups was another Rival touch; 174 were Nexto contacts.

This recovers match contact acquisition from update 100's 4 contacts/0.08 per
minute and introduces measured kickoff first contacts, but still does not show
scoring or sustained possession. The decline in the separately initialized
acquisition probes therefore cannot be treated as a monotonic decline in every
gameplay behavior. Both positive and negative results are retained. This does not
change the user's requested fresh acquisition-only diagnostic.
