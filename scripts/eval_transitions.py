from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tip_env import TripleInvertedPendulumEnv


def run_pair(env: TripleInvertedPendulumEnv, model: SAC, src: int, dst: int, episodes: int, steps: int):
    success = 0
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset(options={"task": (src, dst)})
        ret = 0.0
        ever_ok = False
        for _ in range(steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, _, truncated, info = env.step(action)
            ret += reward
            ever_ok = ever_ok or bool(info.get("is_success", False)) or bool(info.get("ever_success", False))
            if truncated:
                break
        if ever_ok:
            success += 1
        returns.append(ret)
    return success / episodes, float(np.mean(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--xml", default="models/triple_inverted_pendulum.xml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--output", default="outputs/transition_matrix.csv")
    args = parser.parse_args()

    env = TripleInvertedPendulumEnv(
        model_path=args.xml,
        frame_skip=2,
        max_steps=args.steps,
        transition_mode=False,
    )
    model = SAC.load(args.model)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    header = ["src", "dst", "success_rate", "avg_return"]
    lines = [",".join(header)]

    for src in range(8):
        for dst in range(8):
            if src == dst:
                continue
            sr, avg_ret = run_pair(env, model, src, dst, args.episodes, args.steps)
            lines.append(f"{src},{dst},{sr:.4f},{avg_ret:.4f}")
            print(f"{src}->{dst}: success={sr:.3f} return={avg_ret:.1f}")

    out.write_text("\n".join(lines), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
