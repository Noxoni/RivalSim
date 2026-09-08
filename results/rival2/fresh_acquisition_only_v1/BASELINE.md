# Fresh random baseline and first accepted optimizer update

Baseline measured before the first campaign optimizer step using the same fixed
1,024 acquisition probe as the previous mixed run:

- Easy own-contact within 5 seconds: 31/512 = 6.0546875%.
- Varied own-contact within 8 seconds: 11/512 = 2.1484375%.
- Conditional successful-contact medians: 0.508333 s easy, 1.008333 s varied.
- Checkpoint/model/Nexto unchanged during evaluation, zero evaluation steps.

The low conditional times must not obscure the high failure counts. All 1,024
raw first-contact ticks remain in acquisition_000000.json. This is an untrained
baseline, not an improvement claim.

CPU readback proved initialized.pt and plus_000000.pt have identical model tensors
and empty optimizer state. The saved entry confirms a fresh random initialization,
no parent, cleared actor/critic histories, and the frozen acquisition-only bank.
The first accepted PPO update changed the model and saved 136 fresh Adam steps,
4,423,680 learner decisions, and zero exposure from any other scenario family.
No old model or optimizer initializes this lineage.

| Artifact | Raw SHA-256 |
| --- | --- |
| Initialized file | `AB1DE76A547F2C14EA4556E59C55C748BA12B105080CF958D9F4357FEE855EA5` |
| Baseline snapshot | `87B662D84696665701BA3094E1454BD5004169F73AD64852651DB787D8234440` |
| Baseline result | `B0EFAAE15447AABB8A79F4BDDEFD553FD2B25094E1D7BCDF7B81D7EB0FD7C1F5` |
| First accepted snapshot | `4566889FEA453FBF4F310B0679FF87773049360F62B45D39AB005F3D6AA6573C` |

The saved initialization/entry tensor digest is
`38C58096DD2E8B947908892C1A0C6ED0BE46FC0E88A372371172B62BCA18677F`.
The training reset-bank digest is
`C53D80BC6A0725AEA5E1FC68FD9F5144642B75C253931E68B46D12F11022863F`.
The complete baseline CPU audit passed. Prospective settings remain unchanged.
