# OACP Experiment Plan — CoRL 2026 Submission

**Goal:** Reviewer 6/10 → 7/10+
**Review Thread ID:** 019dde15-6a46-7d63-97da-c0d67c5f8287

---

## Instance Allocation Table

| Instance ID | GPU Config | Assigned Runs | Status | Notes |
|---|---|---|---|---|
| oacp2c | 2 × 4090 | D0/D1/D2/D3 | | |
| oacp2 | 2 × 4090 | P1/P2/P3/P4 | | |
| oacp2b | 2 × 4090 | F1/F2/F3/F4 | | |
| oacp2d | 2 × 4090 | A1 (OACP-OV) / A2 (Argmax-OV) | | val_ablation, both open-vocab + 5 holdout |
| | × 4090 | | | |

**Episode 配置：**
- **主实验 (Table 1)：** `val/` full 2195 episodes（已有 baseline/argmax/oacp 各跑了部分）
- **消融实验 (Figure A/B/C)：** `val_ablation/` 330 episodes（11 场景 × 30 ep，seed=42 固定）
- 用法和 `val_mini` 完全一样，eval yaml 改一行：
  ```yaml
  object_nav_path: "datasets/objectnav_mp3d_v1/val_ablation/content/"
  ```
- 每个 330-ep run 约需 **20h**（单 4090）

**并行方案：** 330 episodes 拆成 **5 个 shard**（每份 66 ep），每个 shard 峰值 ~8GB 显存，一块 4090 (49GB) 可同时跑 **5 个 shard**。

| GPU 数量 | 并行策略 | 每个 run 耗时 |
|---|---|---|
| 1× 4090 | 5 个 shard 并行，一次完成 1 个 run | ~4h |
| 2× 4090 | 每 GPU 5 个 shard，一次完成 2 个 run | ~4h |

Shard 数据集：`datasets/objectnav_mp3d_v1/val_ablation_s{0..4}/content/`

**Important:**
1. 所有实验必须在 `tmux` 中运行，防止 SSH 断连。
2. **必须错开启动**：shard 同时启动会在 model tracing 阶段显存峰值叠加导致 OOM。每个 shard 间隔 **120 秒**启动。

3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 必须生效**。已写入 conda env activate hook，启动前验证：
   ```bash
   source /root/miniconda3/etc/profile.d/conda.sh && conda activate onemap
   echo $PYTORCH_CUDA_ALLOC_CONF   # 应输出 expandable_segments:True
   ```
   若不生效则手动设置：
   ```bash
   mkdir -p /root/miniconda3/envs/onemap/etc/conda/activate.d
   echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' > \
     /root/miniconda3/envs/onemap/etc/conda/activate.d/env_vars.sh
   ```

4. tmux 中 `conda activate` 需要先 source init 脚本：
```bash
CONDA_INIT="source /root/miniconda3/etc/profile.d/conda.sh && conda activate onemap"
WD="/inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap"

# 一块 GPU 上并行 5 个 shard（以 run P1 为例），每个 shard 间隔 120s 启动
for s in 0 1 2 3 4; do
  tmux new-session -d -s P1_s${s} \
    "${CONDA_INIT} && cd ${WD} && \
     CUDA_VISIBLE_DEVICES=0 \
     xvfb-run -a python -u eval_habitat.py -c config/mon/eval_oacp_a70.yaml \
     --EvalConf.object_nav_path datasets/objectnav_mp3d_v1/val_ablation_s${s}/content/ \
     --EvalConf.results_path results/mp3d_oacp_a70/s${s} \
     2>&1 | tee results/mp3d_oacp_a70/s${s}/log.txt"
  sleep 120  # 等上一个 shard 完成模型加载
done

# 两块 GPU：GPU 0 跑 P1，GPU 1 跑 P2（各 5 shard）
# 同 GPU 内 shard 必须错开 120s；不同 GPU 的 shard 可同时启动

# 查看所有会话：tmux ls

# 全部结束后合并结果：
mkdir -p results/mp3d_oacp_a70/state
for s in 0 1 2 3 4; do
  cp results/mp3d_oacp_a70/s${s}/state/* results/mp3d_oacp_a70/state/
done
```

---

## 全局配置基线（适用于所有 P/F/C/D 消融）

**所有消融实验统一使用 Open-Vocabulary 设置**，5 个 holdout label 全部屏蔽：

```yaml
clip_cp_open_vocab: True
clip_cp_holdout_labels: "shower,cabinet,chest of drawers,table,tv"
```
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
| `clip_cp_open_vocab` | `self._open_vocab` | 是否空字典启动（消融统一 True） | True |
| `clip_cp_holdout_labels` | `self._holdout_dict` 来源 | open-vocab 屏蔽的初始 label 列表 | `"shower,cabinet,chest of drawers,table,tv"` |
| `clip_cp_calibration_delay` | `self._calibration_delay` | oracle 公布滞后步数 K（同时延迟 expand_label 和 calibrate_aci） | 0 |

**注意：** `self.threshold` 是 **similarity 空间的 τ**，论文中的 nonconformity 阈值 $C_t = 1 - \tau$。τ 越大 → prediction set 越小 → 越保守。

**Delay 机制已实现**（`navigator.py:201-203, 427-428, 806, 897-944`）：oracle 在 t 步观察到 GT cell，将 `(step, cx, cy, label, feat.clone())` 入 `_delay_buffer`；t+K 步出队后**同步**执行 `expand_label`（若是 holdout 首次发现）和 `calibrate_aci`。delay=0 时与改动前行为完全等价。`oracle_noise_rate` 仍在观察时刻判定（oracle 选择不公布则不入队）。

---

## Figure A: Safety-Efficiency Pareto Frontier

**论文呈现：** 一张散点图，x 轴 SCR↓，y 轴 SR↑。图上同时画：
- ◆ OACP (ACI) 的 4 个 coverage target 点，连成 frontier 曲线
- ○ Fixed-threshold CP 的 3 个点，用空心 marker
- ■ Baseline (OneMap) 和 GCLIP Argmax 各 1 个参考点
- 每个点旁标注 STUCK rate

**Story：** OACP-ACI 在不同 coverage target 下形成平滑 frontier；Fixed-CP 散落在 frontier 下方（被 dominated），证明手调固定阈值不如 ACI 自适应。Fixed-CP 只是写了一些示例，具体选多少需要从MP3D这330ep以外再另划一些，离线校准出来具体对应的threshold。

### Runs

| Run ID | Method | Config 改动（相对 `mapping_gclip_oacp.yaml`） | Status |
|---|---|---|---|
| P1 | OACP (ACI) | `clip_cp_target_coverage: 0.70` | [ ] |
| P2 | OACP (ACI) | `clip_cp_target_coverage: 0.80` | [ ] |
| P3 | OACP (ACI) | （当前配置，coverage=0.90） | [ ]  |
| P4 | OACP (ACI) | `clip_cp_target_coverage: 0.95` | [ ] |
| F1 | Fixed-CP | `clip_cp_use_oacp: False`, e.g.`clip_cp_threshold: 0.35` | [ ] |
| F2 | Fixed-CP | `clip_cp_use_oacp: False`, e.g.`clip_cp_threshold: 0.48` | [ ] |
| F3 | Fixed-CP | `clip_cp_use_oacp: False`, e.g.`clip_cp_threshold: 0.55` | [ ] |
| F4 | Fixed-CP |.... 与coverage=0.95对应|

Fixed-CP 的 τ 值预期结果：
- F1 τ=0.35：较松，更多 label 进 prediction set → 检测灵敏但误报多 → 预计 SCR 偏高
- F2 τ=0.48：≈ 当前 ACI 收敛值，看"选对了固定值"能否匹配 adaptive
- F3 τ=0.55：较紧，prediction set 小 → 漏检 → 预计 STUCK 高

参考点（已有，不需要新 run）：
- Baseline (OneMap): SR=26.6%, SCR=26.6%, STUCK=20.3%
- GCLIP Argmax: SR=18.0%, SCR=14.8%, STUCK=25.8%

具体数据结果：

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

| Run ID | delay k (steps) | Config 改动（相对全局基线） | Status |
|---|---|---|---|
| D0 | 0 | `clip_cp_calibration_delay: 0`（open-vocab，与之前 closed-vocab P3 不可直接复用） | [ ] |
| D1 | 5 | `clip_cp_calibration_delay: 5` | [ ] |
| D2 | 10 | `clip_cp_calibration_delay: 10` | [ ] |
| D3 | 20 | `clip_cp_calibration_delay: 20` | [ ] |

注：所有 D-series 都用全局基线（open-vocab + 5 个 holdout），仅 delay 值不同。

**总计：4 new runs**（D0 因切换 open-vocab 重新跑）

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
| **Figure A:** Pareto Frontier | 散点图 | 7 (P1/P2/P3/P4 + F1/F2/F3) — P3 切换 open-vocab 后重跑 | ~20h | ★★★ |
| **Table B:** Closed-set vs OACP | 表格 | 2 (C1/C2) | ~20h | ★★★ |
| **Figure C:** Delay + Convergence | 2×2 折线图 | 4 (D0/D1/D2/D3) | ~20h | ★★☆ |
| **Figure D:** Cold-start Analysis | 柱状图 | 0 | — | ★★ |

**总计：13 new runs × 330 episodes（含 P3 与 D0 重跑）**

所有消融 run 的 eval yaml 使用 `val_ablation` 数据集：
```yaml
  object_nav_path: "datasets/objectnav_mp3d_v1/val_ablation/content/"
```

---

## Execution Checklist

```
Figure A (Pareto, 7 runs — all open-vocab + 5 holdout):
[ ] P1: OACP coverage=0.70 — instance: ______, started: ______, done: ______
[ ] P2: OACP coverage=0.80 — instance: ______, started: ______, done: ______
[ ] P3: OACP coverage=0.90 — instance: ______, started: ______, done: ______ (rerun under open-vocab)
[ ] P4: OACP coverage=0.95 — instance: ______, started: ______, done: ______
[ ] F1: Fixed τ=0.35      — instance: ______, started: ______, done: ______
[ ] F2: Fixed τ=0.48      — instance: ______, started: ______, done: ______
[ ] F3: Fixed τ=0.55      — instance: ______, started: ______, done: ______

Table B (Closed-set, 2 runs):
[ ] C1: Closed-CP-stale   — instance: ______, started: ______, done: ______
[ ] C2: Closed-CP-recomp  — instance: ______, started: ______, done: ______

Figure C (Delay, 4 runs — all open-vocab + 5 holdout):
[ ] D0: delay=0           — instance: oacp2c, started: ______, done: ______
[ ] D1: delay=5           — instance: oacp2c, started: ______, done: ______
[ ] D2: delay=10          — instance: oacp2c, started: ______, done: ______
[ ] D3: delay=20          — instance: oacp2c, started: ______, done: ______

Figure D (Cold-start, 0 runs):
[ ] Analysis from existing logs
```

---

## Implementation Checklist

```
Code changes needed:
[x] 1. Delay buffer in navigator.py (for Figure C: D0/D1/D2/D3) — DONE
       - clip_cp_calibration_delay (default 0) in mapping_conf.py
       - _delay_buffer in __init__/reset/add_data; expand_label & calibrate_aci both deferred K steps
[ ] 2. Softmax score mode in clip_cp_obstacle_map.py (for Table B: C1/C2)
       - New param: use_softmax_score (default False)
       - In calibrate_aci(): s = -log softmax(l_true | V_t) when enabled
       - C2 variant: buffer stores (feat, label), recompute on expand_label()
[x] 3. New config param in mapping_conf.py:
       - clip_cp_calibration_delay: Optional[int] = 0  ✅ DONE
       - clip_cp_use_softmax_score: Optional[bool] = False  (still needed for Table B)

Global config baseline (every P/F/D mapping yaml MUST include):
       clip_cp_open_vocab: True
       clip_cp_holdout_labels: "shower,cabinet,chest of drawers,table,tv"



Result directories:
  results/mp3d_oacp_a70/       results/mp3d_oacp_a80/
  results/mp3d_oacp_a95/       results/mp3d_fixed_cp_t035/
  results/mp3d_fixed_cp_t048/  results/mp3d_fixed_cp_t055/
  results/mp3d_closed_cp_stale/  results/mp3d_closed_cp_recomp/
  results/mp3d_oacp_delay0/    results/mp3d_oacp_delay5/
  results/mp3d_oacp_delay10/   results/mp3d_oacp_delay20/
```

---

## Table X (instance 2d): OACP-OV vs Argmax-OV on val_ablation

Owner: `oacp2d`. Re-run OACP and GCLIP-Argmax on `val_ablation` (330 ep), **both** open-vocab + 5 holdout (`shower, cabinet, chest of drawers, table, tv`) — agent starts with empty obstacle dict, oracle expands via `expand_label()` on first encounter.

| Run ID | Method | Eval yaml | Mapping yaml |
|---|---|---|---|
| A1 | OACP (coverage=0.90) | `config/mon/eval_oacp_ablation.yaml` | `mapping_gclip_oacp.yaml` (already OV) |
| A2 | GCLIP-Argmax (c30)   | `config/mon/eval_argmax_ablation.yaml` | `mapping_gclip_argmax_c30_openvocab.yaml` (new) |

No `.py` changes: `expand_label()` + open-vocab branch already cover both modes (`navigator.py:213-326`); argmax mode no-ops `calibrate_aci` via `not self.use_oacp` early return.

Launch: `bash scripts/launch_2d_series.sh` (10 shards, GPU0=OACP s0..s4, GPU1=argmax s0..s4, 120s stagger, ~4h total).

Results: `results/mp3d_oacp_ablation/`, `results/mp3d_argmax_ablation/`.
