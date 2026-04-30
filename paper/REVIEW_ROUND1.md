# Research Review Summary — CoRL 2026

**Reviewer:** GPT-5.4 via Codex MCP (xhigh reasoning)
**Thread ID:** 019dde15-6a46-7d63-97da-c0d67c5f8287
**Score: 6/10** (borderline) → target 7/10

---

## Key Verdict

实验本身已经是 100% held-out（V₀ = ∅，所有 label 在线发现），但论文初稿没写清楚，导致 Round 1 误判为 4/10。澄清后 → 6/10。

## Remaining Weaknesses

1. **只有一个 operating point**，STUCK=40.6% 太高 → 需要 Pareto sweep
2. **没有 closed-set CP baseline** → 无法证明 pairwise score 优于 softmax score
3. **没有 fixed-threshold 消融** → 无法证明 ACI adaptive 的必要性
4. **没有 calibration 收敛图** → 理论和实验脱节
5. **只有 5 个 obstacle 类别** → 规模偏小

## What Reaches 7/10

- [ ] Pareto frontier 上至少一个可用点（SCR≤10%, SR≥22%, STUCK≤30%）
- [ ] Closed-set CP baseline 在词典扩张后明显更差
- [ ] Calibration figure 验证 ACI 收敛到 target coverage
- [ ] 论文中明确写清 100% held-out cold-start 协议

## What Reaches 8/10 (strong accept)

以上全部 + 更广泛的语义类别 / D_safe 自动化 / 真机实验
