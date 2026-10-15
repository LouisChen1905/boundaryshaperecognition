# 新导航包 topic 与割刀训练标签

支持 `motor_cut`、`cut_ctrl`、`visual_obstacle_front`。原始输入保持不变。

## 回放与探索输入

`iter_data_package()` 在 `Frame` 中保留：

- `visual_obstacle_front`：`ObstacleObservation(timestamp, sensor_id, state, objects)`。
  每个对象保留 `data_type`、`object_id`、`dynamic`、`camera_xyz`。
  严格按外层 `sum`、内层 `sum` 截取 `vObs`、`vInfo`，忽略填充尾部。
- `motor_cut` / `cut_ctrl`：RPM、原始数据时间戳、状态及选用的时间依据。
- `training_label`：实测 RPM 绝对值大于零时为 True；零为 False；未知为 None。

障碍物只与完全相同 `data.timestamp` 的边界点云关联，使用截至该帧
输出前已收到的消息，不使用上一帧障碍物替代缺失数据，也不额外读取未来包。
因此已输出帧之后才到达的障碍物不会回填。该包的抽样中，障碍物包先于
相同时间戳的边界包到达。

`update_exploration()` 把障碍物传入 `BoundaryExplorer.update(...,
visual_obstacle_front=...)`。Explorer 暂存完整观测，并用现有相机外参
与该帧姿态生成 `obstacle_world_xy`，保留相机 Y 分量。缺失观测会清空
当前障碍物。此变更仅准备后续 score 输入；尚未增加障碍物评分、分类解释
或障碍物地图融合，现有安全与目标选择逻辑保持原有行为。

## 时间依据

location 和两种视觉消息的 `data.timestamp` 是毫秒 Unix 时间。
motor_cut 的 `data.timestamp` 属于另一时间域，cut_ctrl 无数据时间戳；
二者明确使用包的 `system_ms`。对于目标时刻，选择已读历史中时间不晚于
目标的最近状态，默认最多保留 0.5 秒，不插值 RPM，不使用未来状态。
这属于接收时间近似对齐，无法从当前格式恢复割刀实际采样的精确 Unix 时间。
不会改变现有点云与位姿仅按采样时间对齐的规则。

历史缓存有界：每种割刀 topic 2048 条、障碍物 128 帧。无效辅助时间戳
跳过并计数；无效有效数量、点坐标或 RPM 会带行号报错。

## 导出可直接用于 goal_score_rl 的目标标签（默认）

```powershell
python navigation_topics.py "D:\Downloads\0047X\navbag_boundary_20261015.jsonl" --output rl_artifacts/motor_goal_labels.jsonl --max-records 6000 --max-labels 20
python goal_score_rl.py prepare --source "D:\Downloads\0047X\navbag_boundary_20261015.jsonl" --labels rl_artifacts/motor_goal_labels.jsonl --output rl_artifacts/motor_dataset.json --max-records 6000 --max-frames 800 --max-labels 20 --max-label-records 20
```

命令行默认 `--label-format goal`；Python API 为 `export_goal_labels()`。
直接重用 `iter_data_package()` 的视觉帧编号、时间、插值位置与姿态，
因此 `frame`、`timestamp`、`robot_pose_world` 可以通过 `prepare` 对齐校验。
新标签类型为 `motor_cut_trajectory_goal`，保留原有手工标签类型支持。
训练数据集 episode 也保留 `label_type` 和 `label_provenance`。

目标配对规则：

- 原点帧与目标帧均满足实测 `abs(motor_cut.rpm) > 0`。
- 从当前帧沿后续对齐轨迹累计移动 `--lookahead-distance` 米（默认 1），
  取首次达到距离且净位移至少 `min(0.2, lookahead_distance)` 米的后续位姿。
  目标朝向使用该处的机器人前向 `head + pi/2`，不使用当前位置充当目标。
- 原点默认按 `--label-interval-s 1` 间隔选取，目标最多等待
  `--max-horizon-s 10` 秒。静止、轨迹不足或到达预算末尾的待配对帧跳过。
- RPM 为零、状态缺失/过期、两帧间隔超过 `--max-frame-gap-s 0.5` 秒、
  重复/不递增时间戳都会结束当前配对段。割刀每个状态包的连续运行起点
  `active_since` 用于识别两帧之间发生的停刀再启动，避免跨段生成目标。
  割刀状态包间隔超过最大状态年龄也会开始新段。

标签带当前帧、目标位姿、两端割刀状态、目标帧号/时间、累计距离、净位移、
姿态对齐方式与全部导出参数。它是记录轨迹上的示范目标，不代表人工最优目标。
未来轨迹仅用于生成训练目标，不加入当前地图观测或候选特征。

默认读取预算为 6000 条原始记录、800 个对齐视觉帧、128 个输出标签；
达到任何预算即停止。待配对队列最多 2048 个小型位姿记录，不保存点云帧。
使用 `--pose-max-gap-s` 时，`prepare` 必须使用相同参数。
输出原子替换，禁止覆盖输入。空标签文件表示当前预算内没有可配对轨迹，
不能交给训练器当作有效数据集。

`prepare` 仍按历史观测重建地图并仅生成安全候选。
存在标签不保证存在可训练 episode；确认其输出的 `episodes` 与 `skipped`
后再运行 `python goal_score_rl.py train --dataset rl_artifacts/motor_dataset.json`。
同一轨迹的前视目标可能跨越时间拆分边界，训练器会排除时间不晚于训练目标
最晚时间的验证帧，并继续保留 `--split-gap-s` 指定的间隔。
若验证集因此为空，应扩大数据预算或增加较晚标签，并用独立日志评估。
本次不自动训练模型，也不添加障碍物 score 项。

## 保留原始训练 location 导出

```powershell
python navigation_topics.py "D:\Downloads\0047X\navbag_boundary_20261015.jsonl" --label-format location --output rl_artifacts/motor_locations.jsonl --max-records 6000
```

Python API `export_motor_labels()` 保持原有行为。
`--motor-max-age-s` 可调整最大状态年龄，默认 0.5 秒。

每条 `motor_cut_active_location` 标签对应一个原始 location，包含原始记录号、
sequence、采样时间、原始 poseOI XYZ/RPY、探索坐标惯例的 XY/yaw
（`head + pi/2`）、实测 motor_cut、cut_ctrl、状态年龄和时间依据。
`abs(motor_cut.rpm) > 0` 是唯一正标签条件，包含负转速。
cut_ctrl 的指令 RPM 不会替代实测 RPM；未知或过期状态不生成标签。

此原始位置格式不包含当前帧到未来目标的配对，不能直接导入 `goal_score_rl.py`。

## 本包抽样结果

当前输入文件名为 `D:\Downloads\0047X\navbag_boundary_20261015.jsonl`。
目标标签验证（2026-10-15）：

- 最大 6000 条输入、800 帧、20 个目标标签；实际读 5502 条、296727021 字节、
  488 帧，导出 20 个目标标签。
- 直接导入 `goal_score_rl.py prepare`，同样预算下读取 5129 条、276100437 字节、
  453 帧，生成 12 个 episode，另 8 个标签因缺少安全正分候选而跳过。
- 示例标签：`rl_artifacts/navbag_boundary_20261015_motor_goal_labels_sample.jsonl`；
  示例数据集：`rl_artifacts/navbag_boundary_20261015_motor_goal_dataset_sample.json`；
  报告：`reports/navbag_boundary_20261015_goal_export_sample.json`、
  `reports/navbag_boundary_20261015_goal_prepare_sample.json`。
- 未训练模型；这些结果验证格式接通与候选集生成，不表示探索效果改善。

之前使用旧文件名进行的原始 location 抽样：

- 标签导出：6000 条原始记录，330201733 字节，3792 个 location；
  正标签 1590，RPM 为零 2195，状态未知 7；跳过 2 条无效辅助时间戳。
- 回放：另读前 800 条，52097249 字节；81 帧对齐，其中全部匹配到障碍物。
  仅前 3 帧送入 Explorer 验证坐标转换，未测量完整 GUI 播放或模型收益。
- 报告：`reports/navbag_boundary_202601015_topic_sample.json`、
  `reports/navbag_boundary_202601015_replay_sample.json`。
- 数据集前缀验证不代表已处理整个 1.55 GB 文件。
