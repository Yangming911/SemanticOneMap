# OACP2Real — OACP Real Robot Deployment

SemanticOneMap OACP 算法的真机部署分支。从主仓库精简而来，只保留核心代码，移除实验/论文/结果等非部署内容。

## 目录结构

```
config/           配置解析 + yaml
eval/             evaluator, actor, dataset utils
mapping/          feature map, navigator, CP obstacle maps, oracle
onemap_utils/     数学/可视化工具
planning/         路径规划 (Python)
planning_cpp/     路径规划 (C++ 加速)
vision_models/    CLIP, GCLIP, YOLO 等视觉模型
eval_habitat.py   仿真评测入口
```

## 核心改动（相对主仓库）

1. **Oracle 接口抽象** (`mapping/oracle.py`)：将 GT label map 查询逻辑抽象为可插拔的 `ObstacleOracle` 接口，便于真机替换为 Grounding DINO / VLM 等外部检测器。
2. **命名规范化**：`holdout_dict` → `oracle_dict`，`clip_cp_holdout_labels` → `clip_cp_oracle_labels`。
3. **移除实验专用逻辑**：`noise_rate`（oracle 噪声消融）、`_oracle_raw_dict`（仅用于日志显示）。

## Oracle 接口（真机部署核心）

真机部署需要实现 `ObstacleOracle` 的子类。接口定义在 `mapping/oracle.py`：

```python
class ObstacleOracle(ABC):

    @abstractmethod
    def detect(self, agent_cell: Tuple[int, int]) -> List[OracleDetection]:
        """返回 agent 当前位置附近的 obstacle 检测结果。

        Args:
            agent_cell: (px, py) agent 在地图 cell 坐标系中的位置。

        Returns:
            List[OracleDetection(label, cell_x, cell_y)]
        """
        ...

    @abstractmethod
    def get_safety_radius(self, label: str) -> Optional[float]:
        """返回某个 label 的初始安全距离（cell 数），未知返回 None。"""
        ...

    def reset(self) -> None:
        """每个 episode 开始时调用。"""
        pass
```

### OracleDetection

```python
@dataclass
class OracleDetection:
    label: str     # obstacle 类别名，如 "table", "chair"
    cell_x: int    # 地图 cell x 坐标
    cell_y: int    # 地图 cell y 坐标
```

### 数据流

```
RGBD 图像
  |
  +---> GCLIP: 像素级 feature --> depth 逆投影 --> 地图 cell 级 feature_map
  |
  +---> Oracle.detect(agent_cell)
          |
          v
        [(label, cell_x, cell_y), ...]
          |
          v
        delay_buffer: (label, cell_xy, clip_feature, threshold)
          |
          v
        两路消费:
          1. 新类别发现 --> expand_label(label, radius, text_feat)
          2. ACI 在线校准 --> calibrate_aci(clip_feat, label)
```

### 已有实现

- `GTLabelMapOracle`：仿真用，从 Habitat GT semantic map 查询。

### 真机实现示例（待开发）

Grounding DINO 实现需要：
1. 输入 RGB 图像 + depth 图像 + 相机位姿
2. DINO 检测得到 bbox + label（像素空间）
3. 用 depth 逆投影将 bbox 中心映射到地图 cell 坐标
4. 返回 `List[OracleDetection]`

```python
class GroundingDINOOracle(ObstacleOracle):

    def detect(self, agent_cell):
        # 1. 用当前帧 RGB 跑 Grounding DINO
        # 2. 过滤低置信度检测
        # 3. bbox 中心 + depth --> 3D --> map cell
        # 4. 返回 OracleDetection 列表
        ...

    def get_safety_radius(self, label):
        # 查找表 or 默认值 (如 0.35m 对应的 cell 数)
        ...
```

### 注意事项

- **校准样本量**：GT oracle 每步查询 agent 周围 3m 半径所有 cell（含历史帧），可产生数百个样本。真机 DINO 仅覆盖当前帧 FOV，样本量小得多，ACI 收敛会更慢。可考虑将 DINO 检测结果累积到一张 detected label map 中来缓解。
- **初始安全距离**：`get_safety_radius()` 在发现新 obstacle 类别时调用，提供 `expand_label` 的初始 radius。可用固定查找表或让 VLM 估计。
- **OACP 参数**：`clip_cp_gamma` 和 `clip_cp_window_size` 可能需要针对真机样本量重新调优。

## 配置

两套评测配置：

| 配置 | 说明 |
|---|---|
| `config/mon/eval_baseline.yaml` | OneMap baseline (无 OACP) |
| `config/mon/eval_oacp.yaml` | OACP (open-vocab, coverage=0.9) |

注入 oracle 方式：
```python
navigator.set_oracle(your_oracle_instance)
```
或通过兼容方法（仿真）：
```python
navigator.set_gt_label_map(label_map, labels)
```
