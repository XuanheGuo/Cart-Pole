from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tip_env import TripleInvertedPendulumEnv

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class LiveTrainingVizCallback(BaseCallback):
    def __init__(
        self,
        run_dir: Path,
        render_freq: int = 50,
        fps: int = 30,
        show_window: bool = False,
        video_path: str | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.render_freq = max(1, int(render_freq))
        self.fps = max(1, int(fps))
        self.show_window = show_window
        self.video_path = video_path

        self.window_available = bool(show_window and cv2 is not None)
        self.start_time = 0.0
        self.last_frame_time = 0.0
        self.episode_returns: deque[float] = deque(maxlen=50)

        self.csv_fp = None
        self.csv_writer = None
        self.video_writer = None
        self.video_disabled_reason = ""

    def _resolve_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.last_frame_time = self.start_time

        metrics_path = self.run_dir / "train_live_metrics.csv"
        self.csv_fp = open(metrics_path, "w", encoding="utf-8", newline="")
        self.csv_writer = csv.DictWriter(
            self.csv_fp,
            fieldnames=[
                "timestep",
                "reward",
                "posture_cost",
                "vel_cost",
                "cart_cost",
                "effort_cost",
                "progress_reward",
                "swing_bonus",
                "stall_penalty",
                "success",
                "source_goal",
                "target_goal",
                "episode_return",
                "episode_length",
                "elapsed_sec",
            ],
        )
        self.csv_writer.writeheader()

        if self.video_path:
            path = Path(self.video_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                self.video_writer = imageio.get_writer(str(path), fps=self.fps)
            except Exception as exc:
                self.video_writer = None
                self.video_disabled_reason = str(exc)
                print(
                    "[LiveTrainingVizCallback] Video recording disabled. "
                    "Install imageio-ffmpeg or use `pip install imageio[ffmpeg]`."
                )
                print(f"[LiveTrainingVizCallback] Backend error: {self.video_disabled_reason}")

    def _overlay(self, frame: np.ndarray, lines: list[str]) -> np.ndarray:
        if cv2 is None:
            return frame
        canvas = frame.copy()
        y = 30
        for line in lines:
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)
            y += 28
        return canvas

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])

        info: dict[str, Any] = infos[0] if infos else {}
        reward = float(rewards[0]) if len(rewards) else 0.0

        episode_return = np.nan
        episode_length = np.nan
        ep = info.get("episode")
        if ep is not None:
            episode_return = float(ep.get("r", np.nan))
            episode_length = float(ep.get("l", np.nan))
            if not np.isnan(episode_return):
                self.episode_returns.append(episode_return)

        elapsed = time.time() - self.start_time
        if self.csv_writer is not None:
            self.csv_writer.writerow(
                {
                    "timestep": int(self.num_timesteps),
                    "reward": reward,
                    "posture_cost": info.get("posture_cost", np.nan),
                    "vel_cost": info.get("vel_cost", np.nan),
                    "cart_cost": info.get("cart_cost", np.nan),
                    "effort_cost": info.get("effort_cost", np.nan),
                    "progress_reward": info.get("progress_reward", np.nan),
                    "swing_bonus": info.get("swing_bonus", np.nan),
                    "stall_penalty": info.get("stall_penalty", np.nan),
                    "success": int(bool(info.get("is_success", False))),
                    "source_goal": info.get("source_goal", -1),
                    "target_goal": info.get("target_goal", -1),
                    "episode_return": episode_return,
                    "episode_length": episode_length,
                    "elapsed_sec": elapsed,
                }
            )

        if self.n_calls % self.render_freq != 0:
            return True

        base_env = self._resolve_base_env()
        frame = base_env.render()

        mean_ret = float(np.mean(self.episode_returns)) if self.episode_returns else np.nan
        sps = int(self.num_timesteps / max(elapsed, 1e-6))
        lines = [
            "Training: SAC",
            f"Timestep: {self.num_timesteps} | SPS: {sps}",
            f"Task: G{info.get('source_goal', '?')} -> G{info.get('target_goal', '?')}",
            f"Reward: {reward:.3f} | Success: {bool(info.get('is_success', False))}",
            f"PostureCost: {float(info.get('posture_cost', np.nan)):.4f}",
            f"AvgEpRet(50): {mean_ret:.2f}",
        ]
        frame = self._overlay(frame, lines)

        if self.video_writer is not None:
            try:
                self.video_writer.append_data(frame)
            except Exception as exc:
                self.video_disabled_reason = str(exc)
                self.video_writer.close()
                self.video_writer = None
                print(
                    "[LiveTrainingVizCallback] Video writer failed during training; "
                    "recording is now disabled."
                )
                print(f"[LiveTrainingVizCallback] Backend error: {self.video_disabled_reason}")

        if self.window_available:
            try:
                cv2.imshow("TIP Training Live", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                key = cv2.waitKey(1)
                if key == 27:
                    return False
            except Exception:
                self.window_available = False

        now = time.time()
        min_dt = 1.0 / self.fps
        dt = now - self.last_frame_time
        if dt < min_dt:
            time.sleep(min_dt - dt)
        self.last_frame_time = time.time()

        return True

    def _on_training_end(self) -> None:
        if self.csv_fp is not None:
            self.csv_fp.flush()
            self.csv_fp.close()
            self.csv_fp = None
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if cv2 is not None:
            cv2.destroyAllWindows()


def build_env(model_path: str, log_dir: Path):
    def _make():
        env = TripleInvertedPendulumEnv(
            model_path=model_path,
            frame_skip=2,
            max_steps=2000,
            transition_mode=True,
            log_dir=str(log_dir / "episodes"),
        )
        return env

    return _make


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/triple_inverted_pendulum.xml")
    parser.add_argument("--total-steps", type=int, default=800_000)
    parser.add_argument("--run-dir", default="outputs/sac_tip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--live-view", action="store_true", help="训练时显示实时仿真窗口")
    parser.add_argument("--live-freq", type=int, default=5, help="每多少个训练step渲染一次")
    parser.add_argument("--live-fps", type=int, default=30)
    parser.add_argument("--train-video", default="", help="训练过程视频输出路径，留空则不保存")
    parser.add_argument("--progress-bar", action="store_true", help="启用SB3进度条(需要rich+tqdm)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([build_env(args.model, run_dir)])
    env = VecMonitor(env)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=300_000,
        batch_size=512,
        gamma=0.99,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        seed=args.seed,
        tensorboard_log=str(run_dir / "tb"),
    )

    ckpt = CheckpointCallback(
        save_freq=25_000,
        save_path=str(run_dir / "checkpoints"),
        name_prefix="sac_tip",
    )
    live_cb = LiveTrainingVizCallback(
        run_dir=run_dir,
        render_freq=args.live_freq,
        fps=args.live_fps,
        show_window=args.live_view,
        video_path=args.train_video if args.train_video else None,
    )

    callbacks = CallbackList([ckpt, live_cb])
    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=args.progress_bar)
    model.save(str(run_dir / "sac_tip_final"))


if __name__ == "__main__":
    main()
