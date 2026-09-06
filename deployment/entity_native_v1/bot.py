"""Separate research bot; never overwrites the installed Rival configuration."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import struct
import sys

import numpy as np
import rlbot.flat as flat
import rlbot.managers
import torch

from runtime import EntityPacketRuntime, ROOT

EXTERNAL=Path("G:/dev/RivalSim-runs/entity-native-comparison-v1")
DECISION=np.dtype([("frame","<i8"),("epoch","<i4"),("resets","<i4"),("team","<i4"),
                   ("missed","<i4"),("history_degraded","u1"),("observation","<f4",(182,)),
                   ("quality","u1",(182,)),("action","<f4",(8,))])
TICK=np.dtype([("frame","<i8"),("active","u1"),("decision","u1"),("action","<f4",(8,))])


def write_json(path, obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)


class EntityBot(rlbot.managers.Bot):
    def __init__(self, candidate, case):
        super().__init__("noxoni/rival/entity-native-"+case)
        self.candidate,self.case=candidate,case
        self.output=EXTERNAL/case
        self.closed=False

    def initialize(self):
        torch.set_num_threads(1)
        m=json.loads((ROOT/f"results/rival2/entity_native_packet_v1/{self.candidate}_manifest.json").read_text())
        self.output.mkdir(parents=True,exist_ok=True)
        if (self.output/"ticks.bin.gz").exists():
            raise RuntimeError("Refuse to overwrite earlier native bot evidence")
        self.runtime=EntityPacketRuntime(m,self.field_info)
        self.decisions=gzip.open(self.output/"decisions.bin.gz","wb",compresslevel=1)
        self.ticks=gzip.open(self.output/"ticks.bin.gz","wb",compresslevel=1)
        self.packets=gzip.open(self.output/"decision_packets.bin.gz","wb",compresslevel=1)
        self.latest_score=None
        self.player_names=None
        write_json(self.output/"ready.json",dict(candidate=self.candidate,case=self.case,
            player_id=self.player_id,team=self.team,source=m["source"],artifact=m["artifact"],
            runtime=self.runtime.summary(),decision_record_bytes=DECISION.itemsize,
            tick_record_bytes=TICK.itemsize, packet_record_format="uint32LE size + RLBot GamePacket.pack bytes; decision packets only"))
        self.progress()

    def progress(self):
        write_json(self.output/"bot_state.json",dict(candidate=self.candidate,case=self.case,
            runtime=self.runtime.summary(),score=self.latest_score,player_names=self.player_names,closed=self.closed))

    def get_output(self, packet):
        if self.closed:
            return flat.ControllerState()
        try:
            if (EXTERNAL/"STOP").exists():
                self.close()
                raise SystemExit(0)
            row=self.runtime.step(packet,self.player_id)
            self.latest_score=[int(t.score) for t in packet.teams]
            self.player_names=[dict(name=p.name,team=int(p.team),player_id=int(p.player_id)) for p in packet.players]
            item=np.zeros(1,dtype=TICK)
            item["frame"]=int(packet.match_info.frame_num)
            item["active"]=self.runtime.last_phase in {"Active","Kickoff"}
            item["decision"]=self.runtime.last_decision is not None
            item["action"]=row
            self.ticks.write(item.tobytes())
            d=self.runtime.last_decision
            if d is not None:
                record=np.zeros(1,dtype=DECISION)
                for key in ("frame","epoch","team","history_degraded","observation","quality","action"):
                    record[key]=d[key]
                record["resets"]=self.runtime.scheduler.stats["resets"]
                record["missed"]=d["missed_ticks_in_interval"]
                self.decisions.write(record.tobytes())
                raw=packet.pack()
                self.packets.write(struct.pack("<I",len(raw))+raw)
                if self.runtime.scheduler.stats["decisions"]%120==0:
                    for stream in (self.decisions,self.ticks,self.packets):
                        stream.flush()
                    self.progress()
            if self.runtime.last_phase=="Ended":
                self.close()
            return flat.ControllerState(throttle=float(row[0]),steer=float(row[1]),pitch=float(row[2]),
                yaw=float(row[3]),roll=float(row[4]),jump=bool(row[5]),boost=bool(row[6]),handbrake=bool(row[7]))
        except Exception as exc:
            write_json(self.output/"failure.json",dict(type=type(exc).__name__,error=str(exc)))
            self.close()
            raise SystemExit(2) from exc

    def close(self):
        if not self.closed:
            self.closed=True
            for stream in (self.decisions,self.ticks,self.packets):
                stream.close()
            self.progress()

    def retire(self):
        self.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",choices=("reference600","finishing650"),required=True)
    p.add_argument("--case",choices=("reference600_blue","reference600_orange","finishing650_blue","finishing650_orange"),required=True)
    a=p.parse_args()
    EntityBot(a.candidate,a.case).run(wants_ball_predictions=False)
