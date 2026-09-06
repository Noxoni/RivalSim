# Exact entry and first accepted learning step

Prospective authority commit `ecead012976418c703990ce4f101d7b8cef19c33` was
pushed and all13 changed files remotely read back before launch. Raw JSON hashes
were checked. The worker then reverified the published package locally.

`entry_integrity.json` is the runner's seven-check entry audit.
`progress_000001.json` is an independent CPU-only26-check audit of actual parent,
entry and first-update checkpoint files plus the first complete training row.
It verifies exact initial model, Adam/groups/counters and four RNG states;
the accepted step changed weights and optimizer counters; preserved PPO/model
contracts; the new reward and corpus are exactly the frozen transition; Nexto
uses native-v5; all model/Adam values are finite. No optimizer is created by the
auditor. The original parent file is unchanged.

Entry SHA: `BE113AB62C2C67735BF1946D0D691D1473E762FB8722C58913F1AAC611578768`.

First accepted SHA: `9387AD6C570CFDBE3D72A45EE3AD31C7566DFEAA078B8F5DDAF60FDCC41C0E4A`.

This is actual learning/resume integrity, **not evidence of gameplay improvement**.
First corrected-controller gameplay evaluation remains+5; final review+15.
Do not call training first-contact payouts standard kickoff competence: some
training starts have prospectively frozen momentum assistance.

Run the CPU audit without interrupting the worker:

```powershell
.venv\Scripts\python.exe -B benchmarks/report_direct_skills_kickoff_race_v1.py 1
```

The same reporter supports completed+5/+15 evaluation artifacts only and requires
complete full-match integrity. It never changes the running campaign.
