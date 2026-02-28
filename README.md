# Triple Inverted Pendulum (MuJoCo + DeepRL)

目标：
- 从**自然下垂**初始状态出发（三级连杆默认稳定在 `q=[0,0,0]`）
- 定义并控制 8 种目标姿态（3 个关节分别取 `{0, pi}`，共 `2^3=8`）
- 支持姿态之间互相转化（`G0..G7` 任意 `src -> dst`）
- 提供实时可视化、算法指示、数据链路日志、视频回放

## 1. 环境准备

```bash
cd /Users/lcq/Desktop/Cart-Pole\ copy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

若需要输出 mp4（训练录像/回放），请确保安装了视频后端：

```bash
pip install imageio-ffmpeg
```

## 2. 训练 DeepRL (SAC) + 训练期实时可视化

建议先做模型可动性检查（不训练）：

```bash
python scripts/smoke_motion.py --show-window --video outputs/smoke_motion.mp4
```

确认小车和三节摆杆连续运动后，再启动训练：

```bash
python scripts/train.py \
  --model models/triple_inverted_pendulum.xml \
  --total-steps 800000 \
  --run-dir outputs/sac_tip \
  --live-view \
  --live-freq 5 \
  --live-fps 30 \
  --train-video outputs/train_live.mp4
```

如需进度条（需要 `rich+tqdm`）可加：

```bash
  --progress-bar
```

训练产物：
- 最终模型：`outputs/sac_tip/sac_tip_final.zip`
- checkpoint：`outputs/sac_tip/checkpoints/`
- tensorboard：`outputs/sac_tip/tb/`
- 逐回合数据链路日志：`outputs/sac_tip/episodes/episode_*.jsonl`
- 训练期实时指标：`outputs/sac_tip/train_live_metrics.csv`
- 训练期视频（可选）：`outputs/train_live.mp4`

训练窗口叠字（训练中实时刷新）：
- `Timestep/SPS`
- `Gsrc -> Gdst`
- `Reward/Success`
- `PostureCost`
- `AvgEpRet(50)`

说明：
- 训练任务默认 `G0(自然下垂) -> G1..G7` 随机目标。
- 若你修改了模型几何/动力学参数，需要重新训练，旧模型不再适配。

## 3. 实时可视化 + 录像

```bash
python scripts/demo_live.py \
  --model outputs/sac_tip/sac_tip_final.zip \
  --xml models/triple_inverted_pendulum.xml \
  --episodes 5 \
  --steps 1800 \
  --fps 50 \
  --show-window \
  --save-video outputs/demo_live.mp4
```

画面叠字（算法指示）：
- `Algo: SAC`
- `Transition: Gx -> Gy`
- `Step/Reward/Success`

## 4. 回放指定日志到视频

```bash
python scripts/replay.py \
  --log logs/live/episode_xxx.jsonl \
  --xml models/triple_inverted_pendulum.xml \
  --output outputs/replay.mp4 \
  --show-window
```

## 5. 8姿态互转评估（全对）

```bash
python scripts/eval_transitions.py \
  --model outputs/sac_tip/sac_tip_final.zip \
  --xml models/triple_inverted_pendulum.xml \
  --episodes 5 \
  --steps 1500 \
  --output outputs/transition_matrix.csv
```

输出：`outputs/transition_matrix.csv`，包括每个 `src->dst` 的成功率和平均回报。

## 6. 数据链路说明

每个 episode 的 `jsonl` 日志包含：
- `reset` 行：`source_goal`, `target_goal`, 初始 `state(qpos,qvel)`
- 每 step 行：
  - `obs`
  - `action`
  - `reward`
  - `info`（成本分解/成功标志/目标姿态）
  - `state(qpos,qvel)`（用于可重放）

## 7. 8种姿态定义

`G0..G7` 对应 3 个关节目标角（`0` 表示下垂，`pi` 表示倒立）：
- `G0 = [0, 0, 0]`
- `G1 = [pi, 0, 0]`
- `G2 = [0, pi, 0]`
- `G3 = [pi, pi, 0]`
- `G4 = [0, 0, pi]`
- `G5 = [pi, 0, pi]`
- `G6 = [0, pi, pi]`
- `G7 = [pi, pi, pi]`

## 8. 文件结构

```text
models/triple_inverted_pendulum.xml   # MuJoCo 模型
src/tip_env.py                        # Gym 环境 + 8姿态任务 + 日志
scripts/train.py                      # SAC 训练
scripts/demo_live.py                  # 实时可视化 + 视频录制
scripts/replay.py                     # 从日志回放视频
scripts/eval_transitions.py           # 8x8互转评估
```
