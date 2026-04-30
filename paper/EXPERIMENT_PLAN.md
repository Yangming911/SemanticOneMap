# OACP Experiment Plan — CoRL 2026 Submission

**Goal:** Reviewer 6/10 → 7/10+
**Review Thread ID:** 019dde15-6a46-7d63-97da-c0d67c5f8287

---

## Instance Allocation Table

| Instance ID | GPU Config | Assigned Runs | Status | Notes |
|---|---|---|---|---|
| | × 4090 | | | |
| | × 4090 | | | |
| | × 4090 | | | |
| | × 4090 | | | |
| | × 4090 | | | |

**Episode 配置：**
- **主实验 (Table 1)：** `val/` full 2195 episodes（已有 baseline/argmax/oacp 各跑了部分）
- **消融实验 (Figure A/B/C)：** `val_ablation/` 330 episodes（11 场景 × 30 ep，seed=42 固定）
- 用法和 `val_mini` 完全一样，eval yaml 改一行：
  ```yaml
  object_nav_path: "datasets/objectnav_mp3d_v1/val_ablation/content/"
  ```
- 每个 330-ep run 约需 **20h**（单 4090）

**并行方案：** 330 episodes 已拆成 6 个 shard（每份 55 ep），每个 shard 占用 ~7GB 显存，一块 4090 可同时跑 **6 个 shard**。

| GPU 数量 | 并行策略 | 每个 run 耗时 |
|---|---|---|
| 1× 4090 | 6 个 shard 并行，一次完成 1 个 run | ~3.5h |
| 2× 4090 | 每 GPU 6 个 shard，一次完成 2 个 run | ~3.5h |

Shard 数据集：`datasets/objectnav_mp3d_v1/val_ablation_s{0..5}/content/`

**Important:** 所有实验必须在 `tmux` 中运行，防止 SSH 断连：
```bash
# 一块 GPU 上并行 6 个 shard（以 run P1 为例）
for s in 0 1 2 3 4 5; do
  tmux new -d -s P1_s${s} \
    "conda activate onemap && \
     CUDA_VISIBLE_DEVICES=0 PYTORCH_NO_NVML=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     xvfb-run -a python -u eval_habitat.py -c config/mon/eval_oacp_a70.yaml \
     --EvalConf.object_nav_path datasets/objectnav_mp3d_v1/val_ablation_s${s}/content/ \
     --EvalConf.results_path results/mp3d_oacp_a70/s${s} \
     2>&1 | tee results/mp3d_oacp_a70/s${s}/log.txt"
done

# 两块 GPU：GPU 0 跑 P1，GPU 1 跑 P2（各 6 shard）
# 只需改 CUDA_VISIBLE_DEVICES=1 和对应 config/results_path

# 查看所有会话：tmux ls
# 全部结束后合并结果：
mkdir -p results/mp3d_oacp_a70/state
for s in 0 1 2 3 4 5; do
  cp results/mp3d_oacp_a70/s${s}/state/* results/mp3d_oacp_a70/state/
done
```

如果 spock CLI override 不工作，可以为每个 shard 复制一份 yaml 并手动改 `object_nav_path` 和 `results_path`。

---

## Code-Level Parameter Reference

代码中的关键参数名及其含义（避免论文符号和代码变量名混淆）：

| Config yaml 参数 | 代码变量 | 含义 | 当前值 |
|---|---|---|---|
| `clip_cp_target_coverage` | `target_coverage` | 目标覆盖率 $1 - \alpha_{\text{target}}$ | 0.9 |
| `clip_cp_threshold` | `self.threshold` | similarity 阈值 **τ**（`sim >= τ` 进 set） | 0.48 |
| `clip_cp_use_oacp` | `self.use_oacp` | 是否启用 ACI 在线更新 | True |
| `clip_cp_gamma` | `self._gamma` | ACI 步长 γ | 0.05 |
| `clip_cp_initial_alpha` | `self._initial_alpha` | α 初始值 | 0.5 |
| `clip_cp_window_size` | `deque(maxlen=...)` | 校准 buffer 滑窗大小 | 200 |
| `clip_cp_max_seeds_per_label` | `self.max_seeds_per_label` | 每类最大 seed 数 | 20 |
| `clip_cp_safety_radius_add` | 构建时 `_radius_add` | 安全半径额外加值（工程 trick） | 1 |
| `clip_cp_use_margin` | `self.use_margin_score` | 是否用 detection-first pipeline | True |
| `clip_cp_open_vocab` | `self._open_vocab` | 是否空字典启动 | True |

**注意：** `self.threshold` 是 **similarity 空间的 τ**，论文中的 nonconformity 阈值 $C_t = 1 - \tau$。τ 越大 → prediction set 越小 → 越保守。

**当前没有实现 delay 机制**——GT label 在 navigator.py line 873 立即使用（robot 在 3m 内直接拿 GT）。需要新增 delay buffer。

---

## Figure A: Safety-Efficiency Pareto Frontier

**论文呈现：** 一张散点图，x 轴 SCR↓，y 轴 SR↑。图上同时画：
- ◆ OACP (ACI) 的 4 个 coverage target 点，连成 frontier 曲线
- ○ Fixed-threshold CP 的 3 个点，用空心 marker
- ■ Baseline (OneMap) 和 GCLIP Argmax 各 1 个参考点
- 每个点旁标注 STUCK rate

**Story：** OACP-ACI 在不同 coverage target 下形成平滑 frontier；Fixed-CP 散落在 frontier 下方（被 dominated），证明手调固定阈值不如 ACI 自适应。

### Runs

| Run ID | Method | Config 改动（相对 `mapping_gclip_oacp.yaml`） | Status |
|---|---|---|---|
| P1 | OACP (ACI) | `clip_cp_target_coverage: 0.70` | [ ] |
| P2 | OACP (ACI) | `clip_cp_target_coverage: 0.80` | [ ] |
| P3 | OACP (ACI) | 不改（当前配置，coverage=0.90） | [x] done |
| P4 | OACP (ACI) | `clip_cp_target_coverage: 0.95` | [ ] |
| F1 | Fixed-CP | `clip_cp_use_oacp: False`, `clip_cp_threshold: 0.35` | [ ] |
| F2 | Fixed-CP | `clip_cp_use_oacp: False`, `clip_cp_threshold: 0.48` | [ ] |
| F3 | Fixed-CP | `clip_cp_use_oacp: False`, `clip_cp_threshold: 0.55` | [ ] |

Fixed-CP 的 τ 值选择逻辑：
- F1 τ=0.35：较松，更多 label 进 prediction set → 检测灵敏但误报多 → 预计 SCR 偏高
- F2 τ=0.48：≈ 当前 ACI 收敛值，看"选对了固定值"能否匹配 adaptive
- F3 τ=0.55：较紧，prediction set 小 → 漏检 → 预计 STUCK 高

参考点（已有，不需要新 run）：
- Baseline (OneMap): SR=26.6%, SCR=26.6%, STUCK=20.3%
- GCLIP Argmax: SR=18.0%, SCR=14.8%, STUCK=25.8%

**总计：6 new runs**

---

## Table B: Closed-Set vs Open-Vocabulary CP

**论文呈现：** 一张表格，同一 coverage target 下三种 score function 的 SR/SCR/STUCK。

| Method | Score | 扩词典时 | SR | SCR | STUCK |
|---|---|---|---|---|---|
| OACP (ours) | `1 - cos(v, ψ(l))` | score 不变 | | | |
| Closed-CP-stale | `-log softmax` | 不重算 buffer | | | |
| Closed-CP-recomp | `-log softmax` | 重算全部 buffer | | | |

**Story：** Closed-CP-stale 扩词典后 softmax 重分配 → 旧 score 失配 → coverage 崩溃。Closed-CP-recomp 虽然重算，但早期小词典的校准数据在大词典下意义变了。OACP 天然不受影响。

### Runs

| Run ID | 代码改动 | Status |
|---|---|---|
| C1 (stale) | `CLIPCPObstacleMap` 新增 `use_softmax_score=True`；`calibrate_aci()` 中 score 改为 `-log softmax`；buffer 照旧存 scalar | [ ] |
| C2 (recomp) | 同 C1，但 buffer 存 `(clip_feat_copy, true_label)`；每次 `expand_label()` 后遍历 buffer 用新 V_t 重算所有 score | [ ] |

使用 Figure A 中 Pareto frontier 上最佳点的 coverage target。

**总计：2 new runs**

---

## Figure C: Delay Ablation + Coverage Convergence

**论文呈现：** 2×2 子图矩阵：
- 上行：empirical coverage rate vs calibration step（4 条线对应 delay k=0,5,10,20）
- 下行：threshold τ vs calibration step（同 4 条线）
- 标注 target coverage 虚线，验证 Proposition 2 预测的收敛行为

**Story：** 验证理论——所有 delay 值最终都收敛到 target coverage，但 delay 越大收敛越慢（transient ∝ kγ）。与 Proposition 2 的 bound 吻合。

### 代码改动（新增 delay buffer）

在 `navigator.py` 的 calibration 循环中，加一个 delay queue：

```python
# navigator.__init__() 新增:
self._calibration_delay = int(getattr(config.mapping, "clip_cp_calibration_delay", 0))
self._delay_buffer = deque()  # (step, cx, cy, true_label, clip_feat)

# navigator.observe() 中，原来直接 calibrate_aci 的地方改为:
self._delay_buffer.append((self._step_count, cx, cy, true_label, _cp_feats[cx,cy,:].clone()))
# 消费 delay 步之前的事件:
while self._delay_buffer and (self._step_count - self._delay_buffer[0][0]) >= self._calibration_delay:
    _, cx_d, cy_d, lbl_d, feat_d = self._delay_buffer.popleft()
    self.clip_cp_obstacle_map.calibrate_aci(feat_d, lbl_d, cell_xy=(cx_d, cy_d))
```

新增 config 参数：`clip_cp_calibration_delay: int = 0`（默认 0 = 无 delay = 当前行为）

### Runs

| Run ID | delay k (steps) | Config 改动 | Status |
|---|---|---|---|
| D0 | 0 | 不改（= P3，当前结果） | [x] done |
| D1 | 5 | `clip_cp_calibration_delay: 5` | [ ] |
| D2 | 10 | `clip_cp_calibration_delay: 10` | [ ] |
| D3 | 20 | `clip_cp_calibration_delay: 20` | [ ] |

**总计：3 new runs**

---

## Figure D: Cold-Start Collision Analysis

**论文呈现：** 堆叠柱状图，按 obstacle 类别分组：
- 蓝色：collision 发生时该类别**尚未被发现**（pre-discovery）
- 红色：collision 发生时该类别**已在字典中**（post-discovery）

**Story：** 大部分 collision 集中在 cold-start 阶段。一旦 expert oracle 发现类别并加入字典，OACP 有效保护。

### 数据来源
从已有 `results/mp3d_oacp_full/collision_causes_*.csv` + `oacp_calibration_*.csv` 交叉分析。

**无需新 run。**

---

## Summary

| 论文呈现 | 类型 | 新 Run 数 | 每 Run 耗时 | 优先级 |
|---|---|---|---|---|
| **Figure A:** Pareto Frontier | 散点图 | 6 (P1/P2/P4 + F1/F2/F3) | ~20h | ★★★ |
| **Table B:** Closed-set vs OACP | 表格 | 2 (C1/C2) | ~20h | ★★★ |
| **Figure C:** Delay + Convergence | 2×2 折线图 | 3 (D1/D2/D3) | ~20h | ★★☆ |
| **Figure D:** Cold-start Analysis | 柱状图 | 0 | — | ★★ |

**总计：11 new runs × 330 episodes，约 220 GPU-hours (单 4090)**

所有消融 run 的 eval yaml 使用 `val_ablation` 数据集：
```yaml
  object_nav_path: "datasets/objectnav_mp3d_v1/val_ablation/content/"
```

---

## Execution Checklist

```
Figure A (Pareto, 6 runs):
[ ] P1: OACP coverage=0.70 — instance: ______, started: ______, done: ______
[ ] P2: OACP coverage=0.80 — instance: ______, started: ______, done: ______
[x] P3: OACP coverage=0.90 — COMPLETED
[ ] P4: OACP coverage=0.95 — instance: ______, started: ______, done: ______
[ ] F1: Fixed τ=0.35      — instance: ______, started: ______, done: ______
[ ] F2: Fixed τ=0.48      — instance: ______, started: ______, done: ______
[ ] F3: Fixed τ=0.55      — instance: ______, started: ______, done: ______

Table B (Closed-set, 2 runs):
[ ] C1: Closed-CP-stale   — instance: ______, started: ______, done: ______
[ ] C2: Closed-CP-recomp  — instance: ______, started: ______, done: ______

Figure C (Delay, 3 runs):
[x] D0: delay=0           — COMPLETED (= P3)
[ ] D1: delay=5           — instance: ______, started: ______, done: ______
[ ] D2: delay=10          — instance: ______, started: ______, done: ______
[ ] D3: delay=20          — instance: ______, started: ______, done: ______

Figure D (Cold-start, 0 runs):
[ ] Analysis from existing logs
```

---

## Implementation Checklist

```
Code changes needed:
[ ] 1. Delay buffer in navigator.py (for Figure C: D1/D2/D3)
       - New config param: clip_cp_calibration_delay (default 0)
       - Delay queue in observe() before calibrate_aci()
[ ] 2. Softmax score mode in clip_cp_obstacle_map.py (for Table B: C1/C2)
       - New param: use_softmax_score (default False)
       - In calibrate_aci(): s = -log softmax(l_true | V_t) when enabled
       - C2 variant: buffer stores (feat, label), recompute on expand_label()
[ ] 3. New config param in mapping_conf.py:
       - clip_cp_calibration_delay: Optional[int] = 0
       - clip_cp_use_softmax_score: Optional[bool] = False

Config files to create:
[ ] config/mon/mapping_gclip_oacp_a70.yaml  (P1)
[ ] config/mon/mapping_gclip_oacp_a80.yaml  (P2)
[ ] config/mon/mapping_gclip_oacp_a95.yaml  (P4)
[ ] config/mon/mapping_gclip_fixed_t035.yaml (F1)
[ ] config/mon/mapping_gclip_fixed_t048.yaml (F2)
[ ] config/mon/mapping_gclip_fixed_t055.yaml (F3)
[ ] config/mon/mapping_gclip_closed_cp.yaml  (C1/C2, flag toggle)
[ ] config/mon/mapping_gclip_oacp_delay5.yaml  (D1)
[ ] config/mon/mapping_gclip_oacp_delay10.yaml (D2)
[ ] config/mon/mapping_gclip_oacp_delay20.yaml (D3)

Result directories:
  results/mp3d_oacp_a70/       results/mp3d_oacp_a80/
  results/mp3d_oacp_a95/       results/mp3d_fixed_cp_t035/
  results/mp3d_fixed_cp_t048/  results/mp3d_fixed_cp_t055/
  results/mp3d_closed_cp_stale/  results/mp3d_closed_cp_recomp/
  results/mp3d_oacp_delay5/    results/mp3d_oacp_delay10/
  results/mp3d_oacp_delay20/
```
