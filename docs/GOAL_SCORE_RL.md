# 目标点 score 强化学习框架

`goal_score_rl.py` 提供有读取预算的数据准备、单步强化学习环境、策略训练、评估和 JSON 模型导出。只使用 NumPy；数据准备复用 `replay_boundary.py`，因此还需要原项目的 Matplotlib。不会修改输入数据或 label 文件。

## 问题定义

- **状态**：按时间对齐、逐帧融合的历史地图，以及机器人当前位姿。训练使用每个安全候选点的几何特征表示该状态下的动作。
- **动作**：已有 explorer 生成的可达、安全、有可见前沿且原始 score 大于零的目标位姿。存在正分边界候选时，动作集只包含边界候选，保留原算法的硬优先级。
- **奖励**：`-min(目标XY距离 / position_scale, 10) - 最短yaw差 / yaw_scale`。默认尺度分别为 1 米、1 弧度。标签只进入奖励，不进入策略特征。
- **终止**：选择一个目标后 episode 立即结束，等价于 `gamma=0` 的 contextual bandit。日志中的下一帧不是所选目标产生的结果，不能把它当成反事实状态转移。

`GoalScoreBanditEnv.reset()` 返回候选特征、baseline logits、action mask；`step(action)` 返回奖励和 `terminated=True`。动作数量随地图变化。它提供 Gym 风格接口，不依赖 Gymnasium，也未声明通过其标准环境检查。

策略学习原有 score 的乘法残差：

```text
baseline = effective_gain / (1 + path_cost + boundary_weight * boundary_distance) - 2 * visits
learned_score = baseline * exp(clip(features @ weights, -8, 8))
pi(action | state) = softmax((log(baseline) + clip(features @ weights, -8, 8)) / temperature)
objective = E_pi[label_reward] - kl_weight * KL(pi || baseline_policy) - l2/2 * ||weights||²
```

使用全动作奖励计算精确期望策略梯度，减少单步任务的采样噪声。默认固定 300 个 epoch；只有训练标签更新参数，验证标签不用于选择 checkpoint。特征包括增益、边界增益占比、路径长度、边界距离、访问次数、机器人坐标下的目标 XY 和相对目标朝向的 sin/cos。尺度来自传感器量程；特征限制在 [-5, 5]。

保留未知地形不可行驶、边界间距、机器人足迹、路径检查、访问惩罚和 goal 保持/刷新机制。模型不能复活原始非正分候选。目标 yaw 仍由现有视野算法产生，本版本学习候选位姿排序，不单独学习 yaw 生成。

## 少量数据运行

以下为已经使用的两标签验证命令；JSONL **行数不是视觉帧数**。准备阶段最多读 12,000 行、680 个对齐帧、6 行 label；选取其中最早的 2 个标签，抵达最后一个所选 label 后停止，不扫描文件尾。

```powershell
python goal_score_rl.py prepare --source "D:\Downloads\0047X\navigation-data-20260922_155147.jsonl" --labels navigation-data-20260922_155147_goal_labels.jsonl --max-records 12000 --max-frames 680 --max-labels 2 --max-label-records 6 --max-candidates 1024
python goal_score_rl.py train
```

输出：

- `rl_artifacts/dataset.json`：配置、读取预算、实际读取量、时间对齐统计、跳过原因及安全候选。
- `rl_artifacts/goal_score_model.json`：特征版本、参数和训练配置。
- `rl_artifacts/training_report.json`：按时间拆分的训练/验证误差、奖励及逐标签结果。

默认准备预算为 15,000 行、800 帧、4 个 label、128 行 label 和每帧 1,024 个候选；没有无上限读包选项。不会读取未标注帧的训练目标，但前面的观测必须逐帧融合以重建地图。`frame + timestamp + source + robot_pose` 必须匹配，错配会报错；预算外和无安全候选的标签会注明原因，不能当成负样本。

配置不同于默认回放时，使用 `--config explorer_config.json` 提供完整或有效的 `ExplorationConfig` 字段；JSON 中必须包含 `robot_radius` 和 `sensor_range`。`--pose-max-gap-s` 和 `--verified-start-radius` 必须与标注时一致。只有确实已有外部验证的草地初始化区域才填写后者。更改原始包位置会触发 source 校验，需要保留对应来源身份的数据与 label。

候选预算按原始 score 降序截断，标签不参与剪枝。报告中的 `candidate_set_truncated` 说明是否截断；截断时评估只是所保留动作集上的结果。`oracle_all_reward` 记录未截断优先级动作集、默认奖励尺度下的最优奖励，用来区分 score 排序问题和候选生成能力上限。

## 接回原算法

```powershell
python replay_boundary.py "D:\Downloads\0047X\navigation-data-20260922_155147.jsonl" --max-records 12000 --goal-score-model rl_artifacts/goal_score_model.json
```

也可以无窗口运行：

```powershell
python replay_boundary.py "D:\Downloads\0047X\navigation-data-20260922_155147.jsonl" --max-records 1200 --profile-only --performance-log rl_artifacts/model_profile.csv --goal-score-model rl_artifacts/goal_score_model.json
```

代码调用：

```python
from boundary_exploration import BoundaryExplorer, ExplorationConfig
from goal_score_rl import GoalScorePolicy

policy = GoalScorePolicy.load("rl_artifacts/goal_score_model.json")
explorer = BoundaryExplorer(ExplorationConfig(robot_radius=0.2, sensor_range=2.0), goal_scorer=policy)
# 后续 update()、trajectory_is_safe() 的调用方式保持一致。
```

不传模型时维持当前算法。传模型时对所有可达候选进行完整搜索，匹配训练时的搜索范围，可能比原来的分阶段提前停止更慢；目标保持机制仍生效，因此每一帧的实际输出不一定等于即时最优排序目标。按模型 metadata 中的配置回放，才能作有效比较。

## 验证边界

按时间排序，默认前 75% 用于训练，剩余为验证；剔除与训练最后时间相距不足 1 秒的验证帧。少于两个可用 label 或间隔筛选后没有验证帧会报错，不会用训练集冒充验证集。单次日志内的时间留出仍高度相关，最终评估应使用独立采集的日志。

报告中的 `baseline_reward` 是相同全搜索动作集上的原始 score 排序，`logged_reward` 是 label 保存的原算法输出，`replay_reward` 是重建回放时原算法的实际输出。后两者可能因原算法分阶段搜索和目标保持而不同于全搜索 baseline。

独立测试包可先用同样预算准备数据，再执行：

```powershell
python goal_score_rl.py evaluate --dataset other_dataset.json --model rl_artifacts/goal_score_model.json --output rl_artifacts/independent_evaluation.json
python -m unittest discover -s tests -q
```

两标签样例只证明完整管线能运行及其标签奖励变化，不能证明真实机器人闭环收益、安全认证或跨场地泛化。要优化长期探索耗时、覆盖率等指标，需要有动作影响下一状态的仿真环境或真实交互数据，再扩展为多步 RL；当前日志没有提供这样的转移依据。

## 本次小样本运行结果

每次准备实际读取原始包前 6,199 行、约 101.53 MB（原始包的 7.37%），得到前 666 个对齐帧；训练标签为 frame 647，时间留出验证标签为 frame 666。它们分别保留 863、864 个有效边界候选，未发生候选截断。验证标签的即时全搜索排序结果：

| 指标 | 原始 score 全搜索 | 学习 score 全搜索 |
|---|---:|---:|
| 与人工目标的 XY 距离 | 0.697 m | 0.243 m |
| 与人工目标的 yaw 差 | 0.502 rad | 0.174 rad |
| 标签奖励 | -1.199 | -0.417 |

该标签中保存的原算法实际输出奖励为 -1.082，不同于表中的全搜索 baseline。两个标签来自同一段近邻静止场景，以上仅是管线验证结果。自动测试共 79 项，包含策略梯度数值校验、角度环绕、读包预算与关闭、标签错配、模型读写、训练/验证隔离以及现有回放与安全算法回归。


## Map-relative boundary direction (version 2)

Fixed robot-relative direction features are replaced by unsigned along/across
boundary distances, confidence-weighted squared parallel alignment, and direction
confidence. The reference line is fitted once per goal search from accumulated
`boundary_hits` near the robot, never from the current observation array. The
closest mapped boundary must be within sensor range. PCA uses a local neighborhood
of radius `min(sensor_range, max(0.3, 4*resolution))`; fewer than three cells,
insufficient spread, or anisotropy below 0.6 produce no direction preference.
Parallel and antiparallel headings are equivalent. Training and learned inference
use parallel view headings if either direction sees frontier gain, otherwise retain
the existing view heading. Goal holding and safety checks still apply. The learned
parallel-alignment weight is constrained nonnegative during training.

Both dataset and model schemas are now version 2. Existing version 1 artifacts
must be regenerated from source replay with `prepare` and then retrained; changing
JSON version numbers or reusing old direction weights is invalid. The checked-in
model and training report have been updated for version 2. Closed-loop
exploration performance still requires validation on separate logs.
