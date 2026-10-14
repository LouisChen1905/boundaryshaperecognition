# 边界形状识别

## 项目结构

- 根目录下的 Python 模块：探索、边界拟合、回放、目标标注和学习。
- `tests/`：回归测试。在项目根目录运行 `python -m unittest discover -s tests -v`。
- `docs/`：[探索功能交接说明](docs/EXPLORATION_HANDOFF.md)、[回放性能说明](docs/REPLAY_PERFORMANCE.md)、[目标评分学习说明](docs/GOAL_SCORE_RL.md)和算法说明。
- `reports/`：已有的性能、时间对齐和拟合诊断 CSV 文件。
- `rl_artifacts/`：已准备的学习数据集、导出的模型和训练报告。
- 原始 `.log`、输入 `.jsonl` 和目标标注 `.jsonl` 文件保留原有路径，以维持回放默认配置和标注来源标识。

本文所有 Python 命令均应在项目根目录运行。已有的历史 CSV 文件现位于 `reports/`；生成新诊断数据时，请明确指定输出路径，例如 `--performance-log reports/exploration_new_profile.csv`。

## 学习得到的探索目标评分

`goal_score_rl.py` 实现了带数据量限制的 JSONL 数据准备、基于标注奖励的单步强化学习环境、NumPy 策略梯度训练、评估和评分模型导出。回放可通过 `--goal-score-model` 加载模型，现有安全约束仍然生效。命令、奖励设计、数据预算和评估限制详见 [GOAL_SCORE_RL.md](docs/GOAL_SCORE_RL.md)。

## 手动标注探索目标

在 JSONL 回放中，按 **S** 或点击 **Skip static**（跳过静止片段）可快进当前静止片段。当首次出现位置相对起始位姿变化超过 3 cm，或三维姿态变化超过 2 度的帧时，回放会暂停。通过 `--static-position-tolerance`（米）和 `--static-angle-tolerance-deg` 可调整阈值。比较以起始位姿为基准，因此累积的小幅运动最终也会结束跳过。每帧仍按时间顺序更新探索器，保留地图观测证据、帧号和已有标注。中间帧不逐帧渲染，也不等待播放延时；处理会定期将控制权交还界面。再次按 **S** 或按 **Esc** 可停止跳过，标注和帧导航操作也会停止跳过。性能 CSV 将此类帧标记为 `display_skipped=1`，其单帧渲染时间为零；定期批量显示更新的耗时在这些行之外测量。向后查看仍受现有的 50 帧历史记录限制。

JSONL 和文本日志回放视图均支持标注期望目标位姿，供训练使用：

```powershell
python replay_boundary.py "navigation-data.jsonl" --max-records 1000 --goal-labels desired_goals.jsonl
```

按 **A** 或点击 **Label goal**（标注目标）暂停播放，在全局 XY 图中点击期望位置，再沿期望前进方向点击一次。第二次点击自动保存位姿，并显示品红色标记和箭头。**Esc** 取消未完成的标注；**Delete** 或 **Delete label**（删除标注）删除当前帧已保存的标注。再次标注同一帧会替换旧标注。切换帧会取消未完成的标注。点击标注前，请关闭绘图工具栏的平移或缩放模式。重新打开同一回放时，会加载已有标注。

默认输出为工作目录中的 `<log stem>_goal_labels.jsonl`，其中 `<log stem>` 为日志文件名去掉扩展名后的部分。文件仅包含手动标注的帧；未标注帧没有训练目标。每行一个 JSON 对象，记录 `source`、从 1 开始的回放帧号 `frame`、传感器时间戳 `timestamp`（秒）、`desired_goal_world`、`desired_goal_robot`、`robot_pose_world`，以及可用时的当前 `algorithm_goal_world`。XY 单位为米；yaw 单位为弧度，从全局 +X 起算，逆时针为正。机器人坐标系中，+X 向前，+Y 向左；机器人前向 yaw 等于日志 yaw 加 pi/2，与回放朝向约定一致。来源文件的绝对路径、帧号和时间戳共同标识一个标注；继续标注时，应使用相同来源位置和帧加载选项。保存的标注是人工指定的期望位姿；探索器不会执行它们，也不会按安全栅格筛选。编辑和删除通过原子重写更新文件，并保留其他标注。

## 仅在草地区域内进行边界探索

`boundary_exploration.py` 是独立的、有状态的栅格探索器。创建 `BoundaryExplorer(ExplorationConfig(robot_radius=..., sensor_range=...))`，再对每个时间对齐后的帧调用 `update(grass_points, boundary_points, (x, y, yaw), timestamp)`。两列点数组表示全局 XY。三列相机 XYZ 数组需要完整的 4×4 `sensor_to_world` 变换，不能丢弃相机 Y 分量。结果包含稀疏的已知草地、边界和安全栅格集合、边界闭合标志、下一个观测位姿及路径，或解释无法提供路径原因的状态。未知栅格隐式表示，始终不可通行。每帧每个栅格的观测证据仅计数一次；草地默认需要两次观测确认，一次边界观测即可阻塞栅格。仅当相邻边界点间距不超过 `boundary_link_gap` 时才连接这些点。

相同时间戳只根据机器人位姿重新评估，不增加证据；`update` 拒绝倒退的时间戳。同一栅格中的边界观测会覆盖草地分类，并清除旧草地证据。默认情况下，该栅格后续连续三次仅观测到草地时，会撤销边界分类（`boundary_clear_observations`）；新边界观测重置此计数，缺失观测不改变计数。之后仅用新观测确认草地。证据不会随时间过期。输入仅区分草地和边界，不区分其他地形类别。

```python
from boundary_exploration import BoundaryExplorer, ExplorationConfig

explorer = BoundaryExplorer(ExplorationConfig(robot_radius=0.2, sensor_range=2.0))
result = explorer.update(grass_world_xy, boundary_world_xy,
                         (robot_x, robot_y, robot_yaw), timestamp_s)
if result.status == "exploring":
    send_goal_to_local_planner(result.goal_pose, result.path_xy)
```

路径规划前会应用机器人的圆形占地范围、位姿裕量和边界安全间距。安全栅格使用栅格方形区域之间的最小距离，保证机器人中心位于该栅格任意位置时都有足够间距；接触未知地形或达到边界安全间距临界值均视为不安全。这会有意缩小狭窄通道的可用空间。路径从机器人实际位姿开始。`trajectory_is_safe` 检查每段直线接触的所有栅格，包括边和角的接触。`command_is_safe` 将独轮车模型圆弧细分，使每段转角不超过 pi/8 弧度，并将圆弧与弦之间的弓高加入占地裕量。一般的局部规划器曲线轨迹需要提供自身的偏差上界；单纯的位姿列表只表示相邻位姿间的直线段。

目标从可达安全栅格中选择，依据是对草地与未知区域交界前沿的观测能力。距已观测边界不超过 `boundary_frontier_distance`（默认 0.3 m，按栅格中心计算）的前沿被视为潜在的未知边界延续。这是邻近性启发式规则，不是对未观测轮廓的预测。在评分为正的观测点中，优先前沿观测点排在普通观测点之前。选择传感器朝向前会检查量程和边界遮挡；角度扫描先最大化视场内的优先前沿收益，再最大化总收益。执行局部规划器提出的运动前，可用 `trajectory_is_safe` 和 `command_is_safe` 检查。

启动时若传感器无法确认机器人下方为草地，返回 `waiting_for_observation`；可通过 `seed_verified_grass_disk` 显式提供经外部验证的初始草地圆盘。位姿本身不能证明地面是草地。算法采用圆形占地范围和有限量程传感器模型；调用方必须提供经标定的尺寸、定位不确定性和相机外参。确认边界闭合时返回 `complete`，表示在栅格洪泛填充检查中，确认的边界栅格包围了机器人。这不代表完整覆盖内部区域，也不代表发现了所有孔洞或不连通区域。边界修正后会重新评估完成状态。轮廓未闭合且无可达安全观测点时返回 `incomplete_unreachable`；可达观测点因访问惩罚或进展超时而耗尽时返回 `stalled`，且不提供目标或路径。

目标搜索使用一次安全栅格最短路径搜索，按可配置的路径半径逐步扩展，默认 1、2、3 m。每阶段通过空间分桶仅对附近前沿评分。存在优先前沿时，只有其可见收益才能使该阶段满足提前停止条件；附近普通观测点不能阻止继续考虑更远的边界观测点。否则使用总收益。当收益达到 `min_visible_gain`（默认 2；每个优先前沿贡献 2，每个普通前沿贡献 1），且评分为正时停止。同一优先级内的评分为：

`gain / (1 + path_length + boundary_distance_weight * boundary_distance) - 2 * visits`

能看到任意优先前沿时，gain 仅计算优先前沿，否则计算全部可见前沿。边界距离从观测点量到已观测边界栅格中心，上限为传感器量程；没有已知边界时忽略此项。距离权重默认 2.0，零表示关闭距离偏好。这样在保留路径代价和重复访问惩罚的同时偏好更近的观测点。优化范围仍受第一个满足条件的路径半径限制，因此不是全局最近边界搜索。所有阶段均不满足条件时，搜索全部可达安全栅格；无正评分优先观测点时，以正评分普通观测点作为备选。不会分配非正评分观测点。在 `ExplorationConfig` 中设置 `path_radii` 和 `min_visible_gain`，或在 JSONL 回放中使用 `--goal-path-radii 1 2 3` 和 `--min-visible-gain 2`。回放还提供 `--boundary-frontier-distance` 和 `--boundary-distance-weight`。性能 CSV 的 `boundary_frontier_cells`、`goal_boundary_gain`、`goal_boundary_distance` 分别记录优先前沿数量、所选可见优先收益和有上限的边界距离（米）。默认值仅为起点，应使用代表性日志标定。

已分配目标在仍可达、安全、评分为正且有进展时保持原位置和朝向，直到下一次刷新。以下任一条件触发新候选选择：距上次选择经过的传感器时间达到 `goal_refresh_interval_s`（2 s），相对上次选择时的位姿移动达到 `goal_refresh_distance`（0.3 m），或经过角度环绕处理后的最短朝向变化达到 `goal_refresh_yaw_deg`（20 度）。每次 `update` 检查阈值；暂停输入不会通过实际时钟计时器刷新。重新评分可能仍选择相同最优栅格，并重新计算朝向。选择同一栅格不重置进展超时，地图变化不推迟周期刷新。CSV 的 `goal_release_reason` 记录 `refresh_interval`、`refresh_distance` 或 `refresh_yaw`，`goal_action` 变为 `selected`。回放选项为 `--goal-refresh-interval-s`、`--goal-refresh-distance`、`--goal-refresh-yaw-deg`。

重新选择前，先按 `goal_reached_distance`（0.15 m，仅检查位置）判断到达；每次分配只增加一次访问计数。新目标必须在到达半径之外，并满足 `goal_spacing`。在 `goal_progress_timeout_s`（15 s）内，距目标的距离若未累计缩短至少 `goal_progress_distance`（0.1 m），则释放目标，并在当前地图版本中排除它。确认草地/边界分类变化时，递增 `map_version`、清除访问惩罚和排除项，并重置进展计时器。未改变分类的额外观测不算进展。新分类出现后，`stalled` 自动重新评估。回放选项为 `--goal-reached-distance`、`--goal-progress-timeout-s`、`--goal-progress-distance`、`--boundary-clear-observations`。

详细拟合步骤和方向约定见[拟合算法说明.md](docs/拟合算法说明.md)。

运行 `python replay_boundary.py` 加载日志，在全局 XY（左图）与相机 XZ（右图）坐标中同步逐帧回放。绿点是实测边界点，蓝色曲线是独立估计的边界趋势，亮橙色箭头和文字表示各紫色曲线中点附近的**跟踪参考方向**。全局图的红色标记和线表示机器人位置与朝向。两个视图共用 Play/Pause（播放/暂停）、Previous/Next（上一帧/下一帧）、滑块或左右方向键。`--speed 2` 以日志记录速率的两倍播放。

JSONL 导航数据包采用实时探索回放。限制读取量检查所提供文件时，运行 `python replay_boundary.py navigation-data.jsonl --max-records 1000`。读取器将 `visual_boundary_points.data.timestamp` 对齐到 `location.data.timestamp`，二者均为毫秒。时间戳完全匹配时使用对应位姿，否则在前后位姿间线性插值位置，并以最短弧四元数 SLERP 插值姿态。完整相机 XYZ 边界与草地点云使用同一对齐位姿。不会以到达顺序、最新位姿、`system_ms` 或外推替代对齐。

两个定位样本间的最大间隔默认 100 ms，可通过 `--pose-max-gap-ms` 或 `iter_data_package(..., pose_max_gap_s=0.1)` 配置。读取器最多保留 2048 个位姿样本和 128 个待处理视觉包。视觉包可在输入记录限制内等待后续样本形成插值区间。无效样本时间戳、过大间隔、早于保留历史的包、队列溢出，以及到文件末尾或记录限制仍未匹配的帧，均跳过并输出诊断。不会为获得插值区间而读取超出 `--max-records` 的行。可选 `alignment_stats` 收集计数；命令行无界面性能分析会打印统计。CSV 的 `pose_alignment`（`exact`/`interpolated`）、`pose_before_timestamp`、`pose_after_timestamp`（秒）和 `pose_gap_ms` 可用于复核所选区间。

回放保留当前地图和最近 50 帧快照。左图显示确认草地（浅绿）、观测边界（红）、机器人中心安全栅格（青绿）、时间对齐后的机器人、可用的选定目标与安全路径，以及原单帧拟合（蓝）。右图显示相机边界和草地点。播放/暂停、上一帧/下一帧及方向键在有限历史范围内工作；向后只显示缓存结果，不重复增加证据。`--speed 2` 将播放速度提高一倍。

回放采用示例值 `--exploration-robot-radius 0.2` 和 `--exploration-sensor-range 2.0`；解释目标或路径前应改为实测值。`--grid-resolution` 控制地图细节。起始草地区域经独立验证后，可用 `--verified-start-radius R` 初始化。未验证机器人下方草地时，即使前方远处可见草地，也会正确报告 `waiting_for_observation`。JSONL 回放显示探索结果，不向机器人发送命令。

定位回放卡顿时，添加 `--performance-log exploration_performance.csv`。每个视觉帧记录 JSONL 读取、坐标变换、栅格化、证据融合、安全栅格计算、闭合检查、目标搜索、绘图对象更新及实际画布渲染耗时（毫秒），以及点和栅格数量。启用日志时用同步画布绘制测量渲染，因此这是诊断运行，播放可能更慢。仅测算法而不绘图时，使用 `--profile-only --max-records 1000 --performance-log exploration_performance.csv`；`artist_update_ms` 和 `render_ms` 此时为空。`callback_ms` 覆盖一次界面帧回调，`update_ms` 仅覆盖 `BoundaryExplorer.update()`。

直接使用 `BoundaryExplorer` 的代码可将空字典以 `performance=timings` 传给 `update()`；调用会填入相同算法阶段耗时与栅格计数，不写文件。

一帧含 N 个输入点、G 个确认草地栅格、B 个映射边界栅格、S 个安全栅格、F 个草地前沿和 K 个占地偏移时，栅格化约为 O(N)，安全检查为 O(GK)，闭合洪泛填充为 O(A)，A 为有限地图矩形范围的大小。机器人在安全栅格上时，可达路径搜索为 O(S log S)；目标评分可能为 O(FB + SF L/r)，其中 L 为视线长度，r 为栅格分辨率。随着探索面积扩大，最后一项可能占主导，因为大量候选观测点反复检查前沿可见性。未知或不安全起点完全跳过目标搜索。

可见性检查遇到第一个阻塞栅格就停止；映射边界不变时跨帧复用结果。添加或撤销边界使缓存失效，缓存最多 100,000 个观测点/前沿配对。CSV 的 `sight_checks`、`sight_cache_hits`、`goal_search_radius`、`goal_visible_gain` 支持搜索开销分析和阈值标定。验证当前目标而非新选目标时，`goal_search_radius` 为 `held`。其他字段 `map_version`、`observation_fused`、`removed_boundary_cells`、`goal_action`（`selected`、`held` 或 `none`）、`goal_release_reason`、`goal_visit_count` 描述证据和目标生命周期变化；访问计数属于当前目标，无目标时为零。重复帧仍显示和记录但不融合；倒退的视觉时间戳及无效样本时间戳会被跳过。

紫色虚线是两个视图中的同一跟踪参考线。它在全局 XY 中计算，将每个拟合边界点沿局部法线朝草坪侧移动 0.4 m，再调整点序使边界位于跟踪方向的**右侧**。闭合草坪外边界上，这对应逆时针行驶、草坪在左。有草地点时据此确定草坪侧，否则暂以机器人位置估计内部侧，并在图中注明。弯曲过急、不能平滑向内偏移 0.4 m 的曲线退回到更简单的趋势。位置和 yaw 几乎不变的连续帧一起拟合，保持静止时参考线稳定。全局参考随后通过该帧位姿与相机外参映射到相机 XZ，与日志点一样假设相机 Y=0；并非对相机 XZ 拟合另行偏移 0.4 m。

图标题分别报告记录位姿的环行方向及采样参考中点的行进方向。摘要要求路线近似闭合；起终点间距过大时报告“open or incomplete”（未闭合或不完整）。有符号面积为正且位姿 yaw 增大表示全局 XY 中逆时针（CCW）。几何摘要不能替代草地覆盖或机器人行驶极限验证。

日志每行可在 `boundary_xz` 后、`location info` 前添加草地点，格式为 `grass_xz: [(x, z); (x, z)]`。或传入 `--grass-csv grass.csv`，列为 `frame,x,y`（全局 XY）或 `frame,camera_x,camera_z`（相机 XZ），`frame` 从 1 开始。草地点云为浅绿。至少五个点且 70% 在同一侧，才能覆盖基于机器人位置的侧向估计。草地点足够时，显示位于有向参考线左侧的点比例，以及 0.25 m 内有草地点的参考采样点比例。按实际点云密度设置 `--grass-support-radius`。这些是辅助观测，不能证明所有参考点都在草地内。示例日志没有草地点，无法验证参考是否在草坪内。

已知最小转弯半径时，设置 `--min-turn-radius R`。拟合曲率上限为 `1 / (R + 0.4)` m⁻¹，同时保留原有 1.5 m⁻¹ 上限。这是几何转弯半径约束，不模拟转向瞬态、速度或障碍物。未提供 `R` 时，仅凭点云不能保证可跟踪性。

设置 `--robot-width W`，报告跟踪时机器人右侧车身边缘到拟合边界的标称距离 `0.4 - W/2` 米。不对称机器人改用 `--right-half-width H`，计算 `0.4 - H`。选项互斥，非正值以红色突出。计算假设参考跟踪机器人中心，忽略位姿误差、车身形状和转弯时角部扫掠。提供的日志未指定机器人宽度。

每帧的 `boundary_shape.recognize(points_xyz)` 根据全局边界点估计中心线：

1. 用主成分分析找点云长轴。
2. 沿长轴分箱，对有点的分箱取横向坐标中位数。
3. 剔除横向离群点，在曲率限制下，使用分箱中位数拟合三次、二次或线性趋势。
4. 计算切线方向、长度、净转角和粗略形状标签：直线、左弯、右弯或 S 弯。

`orient_shape(shape, robot_xy)` 从最靠近机器人的端点开始排列蓝色中心线；左/右形状标签以此顺序为准。橙色跟踪箭头遵循独立的边界在右侧规则，可能与蓝色中心线顺序相反。全局图方向角从全局 +X 起算、逆时针为正；相机图从相机 +X 向 +Z 起算。点太少或跨度太小的帧返回 `None`，显示“insufficient shape data”（形状数据不足）。

分析是对单条、基本连续边界的几何近似。分支、环路、相距较远的多条边界及严重遮挡可能使单一趋势误导。标签和阈值尚未经人工标注真值数据标定。
