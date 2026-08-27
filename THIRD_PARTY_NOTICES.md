# Third-party notices and source custody

RivalSim v0.1 and v0.2 use source-backed constants, equations, state transitions, vehicle
parameters, and operation ordering inspected in RocketSim and its modified Bullet vehicle
path. The reference source was kept outside the tracked RivalSim tree under `.reference/`; no
RocketSim or Bullet source file was copied wholesale into this repository.

The source revisions used for the v0.1 and v0.2 evidence are:

- primary RocketSim source: `ZealanL/RocketSim` commit
  `c2baacb8f4b441dd8505e63c2aeb5a1679b60b02`;
- Python binding source/package lineage: `mtheall/RocketSim` commit
  `2da51b1dac7b8127127613a5ff30e490bdd70dd8`, released as `rocketsim==2.2.1`;
- installed `RocketSim.pyd` SHA-256:
  `E3EE24CA82445B4BFCC754583F6778D7B0D8B7A7F7D64F872BE8C65E621A63D0`.

RivalSim v0.2 also reads external Rocket League Soccar collision files from `Noxoni/Rival`
commit `36cb14cf645c4f06b668c34d85ce1a500e4b53da`; those files were introduced there by
`4f2b21c00e2fcb7108ab1006fd950b066fbd0484`. They remain local inputs. No raw, extracted, or
repacked `.cmf` geometry is distributed by RivalSim; only loader code, hashes, counts, bounds,
and provenance are committed.

## RocketSim

MIT License

Copyright (c) 2022 ZealanL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Bullet Physics

RocketSim uses Bullet. The v0.1 integrator follows the relevant semi-implicit and damping
order observed through that source lineage, while v0.2 derives wheel/suspension ordering and
parameters from RocketSim's modified `btVehicleRL` path and implements an independently
written bounded static-contact approximation. RivalSim does not claim solver equivalence;
the published v0.2 parity evidence records the remaining divergence.

The files in this repository are licensed under the zlib license, except for the files under
`Extras` and `examples/ThirdPartyLibs`.

Bullet Continuous Collision Detection and Physics Library

<http://bulletphysics.org>

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this
software.

Permission is granted to anyone to use this software for any purpose, including commercial
applications, and to alter it and redistribute it freely, subject to the following
restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote
   the original software. If you use this software in a product, an acknowledgment in the
   product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as
   being the original software.
3. This notice may not be removed or altered from any source distribution.

## Wisp v2-75B

The frozen opponent integration under `third_party/wisp75b/` uses the public
Wisp v2-75B policy and shared-head artifacts retrieved through the RLBot v5
BotPack submodule. BotPack pins `NicEastvillage/RLBot-Wisp-v2-py` commit
`58d4ab18fd0c92529b5ae6582ecf1713a6b1887a`. Exact artifact paths, Git blob
identities, byte SHA-256 values, and the BotPack retrieval commit are recorded
in `third_party/wisp75b/PROVENANCE.json`. The policy weights are unmodified.

Wisp v2-75B is distributed under the MIT License. The complete upstream notice
is retained at `third_party/wisp75b/LICENSE`.
