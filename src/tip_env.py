from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


@dataclass
class TransitionTask:
    source_goal: int
    target_goal: int


class TripleInvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(
        self,
        model_path: str,
        frame_skip: int = 4,
        max_steps: int = 2000,
        transition_mode: bool = True,
        log_dir: str | None = None,
        seed: int | None = None,
        render_width: int = 960,
        render_height: int = 720,
        render_camera: str = "side_2d",
    ) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.render_width = render_width
        self.render_height = render_height
        self.renderer = None
        self.render_ready = False

        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.transition_mode = transition_mode
        self.render_camera = render_camera
        self.np_random = np.random.default_rng(seed)

        self.goal_table = self._build_goal_table()
        self.task = TransitionTask(source_goal=0, target_goal=7)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_dim = 2 + 6 + 3 + 3 + 1 + 8
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.step_count = 0
        self.last_info: dict[str, Any] = {}
        self.last_action = 0.0
        self.prev_posture_cost = 0.0

        self.log_dir = Path(log_dir) if log_dir else None
        self._log_fp = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.render_ready = False
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None

    def _ensure_renderer(self) -> bool:
        if self.render_ready and self.renderer is not None:
            return True
        try:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )
            self.render_ready = True
            return True
        except Exception:
            try:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
                self.render_ready = True
                return True
            except Exception:
                self.renderer = None
                self.render_ready = False
                return False

    def _build_goal_table(self) -> np.ndarray:
        goals = []
        # 8种姿态: 每个关节目标角在 {0, pi}
        for g in range(8):
            bits = [(g >> k) & 1 for k in range(3)]
            target = np.array([math.pi if b else 0.0 for b in bits], dtype=np.float32)
            goals.append(target)
        return np.stack(goals, axis=0)

    def set_task(self, source_goal: int, target_goal: int) -> None:
        self.task = TransitionTask(source_goal=source_goal % 8, target_goal=target_goal % 8)

    def sample_task(self) -> TransitionTask:
        # 训练默认从自然下垂姿态 G0 出发，目标在其余7个姿态中采样
        src = 0
        dst = int(self.np_random.integers(0, 8))
        while dst == src:
            dst = int(self.np_random.integers(0, 8))
        self.task = TransitionTask(src, dst)
        return self.task

    def _joint_angles(self) -> np.ndarray:
        return self.data.qpos[1:4].copy()

    def _joint_velocities(self) -> np.ndarray:
        return self.data.qvel[1:4].copy()

    def _cart_state(self) -> tuple[float, float]:
        return float(self.data.qpos[0]), float(self.data.qvel[0])

    def _goal_onehot(self, goal_idx: int) -> np.ndarray:
        onehot = np.zeros(8, dtype=np.float32)
        onehot[goal_idx] = 1.0
        return onehot

    def _target(self) -> np.ndarray:
        return self.goal_table[self.task.target_goal]

    def _source(self) -> np.ndarray:
        return self.goal_table[self.task.source_goal]

    def _angle_error(self, q: np.ndarray, target: np.ndarray) -> np.ndarray:
        return wrap_to_pi(q - target)

    def _reward(self, action: np.ndarray) -> tuple[float, dict[str, float]]:
        q = self._joint_angles()
        qd = self._joint_velocities()
        x, xd = self._cart_state()
        target = self._target()
        angle_err = self._angle_error(q, target)

        posture_cost = float(np.sum(angle_err**2))
        vel_cost = float(0.03 * np.sum(qd**2))
        cart_cost = float(0.12 * x**2 + 0.02 * xd**2)
        effort_cost = float(0.002 * np.sum(action**2))
        edge_penalty = float(0.0)
        swing_bonus = float(0.0)
        stall_penalty = float(0.0)
        progress_reward = float(0.0)
        abs_x = abs(x)
        if abs_x > 4.5:
            edge_penalty = 0.3 * (abs_x - 4.5) ** 2

        # Encourage monotonic progress toward target posture.
        progress_reward = 1.2 * (self.prev_posture_cost - posture_cost)
        # Encourage active swing-up when far from target to avoid passive center hovering.
        if posture_cost > 1.2:
            swing_bonus = 0.08 * min(abs(xd), 4.0) + 0.04 * min(np.linalg.norm(qd), 10.0)
            if abs_x < 0.35 and abs(xd) < 0.12 and np.linalg.norm(qd) < 0.2:
                stall_penalty = 0.4

        reward = 3.5 - (2.5 * posture_cost + vel_cost + cart_cost + effort_cost + edge_penalty)
        reward += progress_reward + swing_bonus - stall_penalty

        comps = {
            "posture_cost": posture_cost,
            "vel_cost": vel_cost,
            "cart_cost": cart_cost,
            "effort_cost": effort_cost,
            "edge_penalty": edge_penalty,
            "progress_reward": progress_reward,
            "swing_bonus": swing_bonus,
            "stall_penalty": stall_penalty,
            "reward": reward,
        }
        return reward, comps

    def _success(self) -> bool:
        q = self._joint_angles()
        err = np.abs(self._angle_error(q, self._target()))
        return bool(np.all(err < 0.15) and np.linalg.norm(self._joint_velocities()) < 0.35)

    def _obs(self) -> np.ndarray:
        x, xd = self._cart_state()
        q = self._joint_angles()
        qd = self._joint_velocities()
        target = self._target()
        err = self._angle_error(q, target)
        goal_id = np.array([self.task.target_goal / 7.0], dtype=np.float32)
        obs = np.concatenate([
            np.array([x, xd], dtype=np.float32),
            np.sin(q),
            np.cos(q),
            qd,
            err,
            goal_id,
            self._goal_onehot(self.task.target_goal),
        ]).astype(np.float32)
        return obs

    def get_state(self) -> dict[str, list[float]]:
        return {
            "qpos": self.data.qpos.copy().tolist(),
            "qvel": self.data.qvel.copy().tolist(),
        }

    def _set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _init_from_source_pose(self) -> None:
        qpos = np.zeros(self.model.nq, dtype=np.float64)
        qvel = np.zeros(self.model.nv, dtype=np.float64)

        src = self._source()
        qpos[1:4] = src + self.np_random.normal(0.0, 0.05, size=3)
        qpos[0] = float(self.np_random.normal(0.0, 0.02))

        qvel[:4] = self.np_random.normal(0.0, 0.01, size=4)
        self._set_state(qpos, qvel)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if options and "task" in options:
            src, dst = options["task"]
            self.set_task(int(src), int(dst))
        elif self.transition_mode:
            self.sample_task()
        else:
            self.set_task(0, 7)

        self._init_from_source_pose()

        self.step_count = 0
        self.last_action = 0.0
        q = self._joint_angles()
        self.prev_posture_cost = float(np.sum(self._angle_error(q, self._target()) ** 2))

        if self.log_dir:
            if self._log_fp is not None:
                self._log_fp.close()
            ts = int(time.time() * 1000)
            self._log_fp = open(self.log_dir / f"episode_{ts}.jsonl", "w", encoding="utf-8")
            reset_payload = {
                "event": "reset",
                "source_goal": self.task.source_goal,
                "target_goal": self.task.target_goal,
                "state": self.get_state(),
            }
            self._log_fp.write(json.dumps(reset_payload, ensure_ascii=False) + "\n")

        info = {
            "source_goal": self.task.source_goal,
            "target_goal": self.task.target_goal,
        }
        self.last_info = info
        return self._obs(), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        action = np.clip(action, -1.0, 1.0)
        filtered_action = 0.3 * self.last_action + 0.7 * float(action[0])
        self.last_action = filtered_action

        self.data.ctrl[0] = filtered_action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        reward, reward_components = self._reward(action)
        self.prev_posture_cost = float(reward_components["posture_cost"])
        success = self._success()

        terminated = False
        x, _ = self._cart_state()
        out_of_track = abs(x) > 4.8
        truncated = self.step_count >= self.max_steps or out_of_track

        obs = self._obs()
        info = {
            **reward_components,
            "is_success": success,
            "source_goal": self.task.source_goal,
            "target_goal": self.task.target_goal,
            "step": self.step_count,
            "out_of_track": out_of_track,
        }
        self.last_info = info

        if self._log_fp is not None:
            payload = {
                "t": self.step_count,
                "obs": obs.tolist(),
                "action": action.tolist(),
                "reward": reward,
                "info": info,
                "state": self.get_state(),
            }
            self._log_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return obs, reward, terminated, truncated, info

    def render(self):
        if not self._ensure_renderer():
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
        try:
            self.renderer.update_scene(self.data, camera=self.render_camera)
        except Exception:
            self.renderer.update_scene(self.data)
        return self.renderer.render()
