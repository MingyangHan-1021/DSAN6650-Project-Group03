"""
Episode logger: saves one episode as CSV / JSON files for analysis and replay.

Output folder:
    events.csv     one row per event: t, agent, event, station, detail
                   (take, chop, place, handoff, deliver, order completed / expired, burnt, ...)
    actions.csv    one row per step: t, both agents' actions, reward
    frames.jsonl   one JSON object per step: full state snapshot, for replay
    summary.json   episode summary metrics

Usage:
    logger = EpisodeLogger(core)
    core.reset(seed)
    logger.start()
    while not done:
        _, r, _, done, info = core.step(actions)
        logger.record(actions, r, info)
    logger.save("results/logs/run1")
"""

from __future__ import annotations

import csv
import json
import os


class EpisodeLogger:
    def __init__(self, core, save_frames: bool = True):
        self.core = core
        self.save_frames = save_frames
        self.start()

    def start(self):
        """Call after core.reset()."""
        self.events = list(self.core.step_log)   # orders that arrived at reset
        self.actions = []
        self.frames = [self.core.snapshot()] if self.save_frames else []
        self.summary = {}

    def record(self, actions, reward: float, info: dict):
        t = info["t"] - 1
        names = self.core.layout.names
        desc = ["wait" if a == 0 else names[a - 1] for a in (int(actions[0]), int(actions[1]))]
        self.actions.append({"t": t, "a0": int(actions[0]), "a0_target": desc[0],
                             "a1": int(actions[1]), "a1_target": desc[1], "reward": reward})
        self.events.extend(info["log"])
        if self.save_frames:
            self.frames.append(self.core.snapshot())
        self.summary = {k: v for k, v in info.items() if k not in ("events", "log")}

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "events.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "agent", "event", "station", "detail"])
            w.writeheader()
            w.writerows(self.events)
        with open(os.path.join(out_dir, "actions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["t", "a0", "a0_target", "a1", "a1_target", "reward"])
            w.writeheader()
            w.writerows(self.actions)
        if self.save_frames:
            with open(os.path.join(out_dir, "frames.jsonl"), "w", encoding="utf-8") as f:
                for fr in self.frames:
                    f.write(json.dumps(fr) + "\n")
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
