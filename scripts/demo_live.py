from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tip_env import TripleInvertedPendulumEnv

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


GOAL_NAMES = {
    0: "G0(000)",
    1: "G1(001)",
    2: "G2(010)",
    3: "G3(011)",
    4: "G4(100)",
    5: "G5(101)",
    6: "G6(110)",
    7: "G7(111)",
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SAC model zip path")
    parser.add_argument("--xml", default="models/triple_inverted_pendulum.xml")
    parser.add_argument("--mj-model", dest="xml", help="Alias of --xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--save-video", default="outputs/demo_live.mp4")
    parser.add_argument("--record-video", dest="save_video", help="Alias of --save-video")
    parser.add_argument("--source-goal", type=int, default=0)
    parser.add_argument("--target-goal", type=int, default=7)
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()

    env = TripleInvertedPendulumEnv(
        model_path=args.xml,
        frame_skip=2,
        max_steps=args.steps,
        transition_mode=False,
        log_dir="logs/live",
    )

    agent = SAC.load(args.model)

    video_path = Path(args.save_video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=args.fps)

    for ep in range(args.episodes):
        obs, info = env.reset(options={"task": (args.source_goal, args.target_goal)})
        ep_reward = 0.0
        src = info["source_goal"]
        dst = info["target_goal"]

        for t in range(args.steps):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, _, truncated, info = env.step(action)
            ep_reward += reward

            frame = env.render()
            frame = overlay_text(
                frame,
                [
                    f"Algo: SAC",
                    f"Transition: {GOAL_NAMES[src]} -> {GOAL_NAMES[dst]}",
                    f"Step: {t+1}  Reward: {reward:.3f}",
                    f"Success: {info['is_success']}",
                ],
            )
            writer.append_data(frame)

            if args.show_window and cv2 is not None:
                cv2.imshow("TIP Live", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) == 27:
                    truncated = True

            if truncated:
                break
            time.sleep(max(0.0, 1.0 / args.fps))

        print(f"Episode {ep} | {GOAL_NAMES[src]} -> {GOAL_NAMES[dst]} | return={ep_reward:.2f}")

    writer.close()
    env.close()
    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
