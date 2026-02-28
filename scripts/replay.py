from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tip_env import TripleInvertedPendulumEnv

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if cv2 is None:
        return frame
    canvas = frame.copy()
    y = 32
    for line in lines:
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2, cv2.LINE_AA)
        y += 30
    return canvas


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="episode_*.jsonl")
    parser.add_argument("--xml", default="models/triple_inverted_pendulum.xml")
    parser.add_argument("--output", default="outputs/replay.mp4")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.log))
    if not rows:
        raise RuntimeError("Empty log file")

    env = TripleInvertedPendulumEnv(
        model_path=args.xml,
        frame_skip=1,
        max_steps=100,
        transition_mode=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=args.fps)

    source_goal, target_goal = 0, 7
    for row in rows:
        if row.get("event") == "reset":
            source_goal = int(row.get("source_goal", 0))
            target_goal = int(row.get("target_goal", 7))
            env.set_task(source_goal, target_goal)

        state = row.get("state")
        if not state:
            continue

        qpos = np.asarray(state["qpos"], dtype=np.float64)
        qvel = np.asarray(state["qvel"], dtype=np.float64)
        env._set_state(qpos, qvel)

        step = int(row.get("t", 0))
        reward = float(row.get("reward", 0.0))

        frame = env.render()
        frame = overlay_text(
            frame,
            [
                "Replay",
                f"Transition: G{source_goal} -> G{target_goal}",
                f"Step: {step}",
                f"Reward: {reward:.3f}",
            ],
        )

        writer.append_data(frame)
        if args.show_window and cv2 is not None:
            cv2.imshow("TIP Replay", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:
                break

    writer.close()
    env.close()
    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
