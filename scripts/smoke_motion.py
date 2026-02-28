from __future__ import annotations

import argparse
import math
import sys
import time
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default="models/triple_inverted_pendulum.xml")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--video", default="outputs/smoke_motion.mp4")
    args = parser.parse_args()

    env = TripleInvertedPendulumEnv(
        model_path=args.xml,
        frame_skip=2,
        max_steps=int(args.seconds * args.fps * 2),
        transition_mode=False,
    )
    obs, info = env.reset(options={"task": (0, 7)})
    print("task", info)

    writer = None
    if args.video:
        out = Path(args.video)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            writer = imageio.get_writer(str(out), fps=args.fps)
        except Exception as exc:
            print("video disabled:", exc)

    total_steps = int(args.seconds * args.fps)
    for i in range(total_steps):
        t = i / args.fps
        # 开环正弦+二次谐波，确保小车/摆杆明显运动
        u = 0.8 * math.sin(1.5 * t) + 0.30 * math.sin(3.0 * t)
        u = float(np.clip(u, -1.0, 1.0))
        obs, reward, _, trunc, step_info = env.step(np.array([u], dtype=np.float32))

        frame = env.render()
        if cv2 is not None:
            txt = f"t={t:5.2f}s  u={u:+.2f}  x={obs[0]:+.2f}"
            cv2.putText(frame, txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 1, cv2.LINE_AA)

        if writer is not None:
            writer.append_data(frame)

        if args.show_window and cv2 is not None:
            cv2.imshow("TIP Motion Smoke", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:
                break

        if trunc:
            break
        time.sleep(max(0.0, 1.0 / args.fps))

    if writer is not None:
        writer.close()
    env.close()
    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
