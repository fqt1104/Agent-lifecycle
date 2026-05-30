# Agent 生命周期演示 Demo

基于 "方向B：Agent 角色与记忆系统 RFC" 的可视化演示，使用 Streamlit 构建。展示多个 Agent 在任务市场竞争中，从白板状态逐步积累经验、提炼技能、形成 Persona，最终分裂或退休的完整生命周期。

## 快速启动

```bash
cd agent_demo
pip install -r requirements.txt
streamlit run main.py
```

## 项目结构

```
agent_demo/
├── agent_model.py       # 数据模型: Agent, Experience, Skill, Persona, Task 等
├── state_machine.py     # 状态机核心: 10 个状态 + 13 条转移规则
├── mock_data.py         # Mock 层: 向量检索/LLM 反思/分裂/退休的模拟实现
├── task_market.py       # 任务市场: Boss 发布任务 + Agent 加权抽签竞标
├── sim_engine.py        # 模拟引擎: 每个 tick 驱动一轮完整生命周期
├── main.py              # Streamlit UI: 控制面板 + 状态图 + Agent 卡片 + 事件日志
├── test_verify.py       # 核心逻辑验证脚本
└── requirements.txt     # 依赖: streamlit
```

## 核心概念

### Agent 生命周期

Agent 从创建到消亡经历 10 个状态，由状态机强制按序流转：

```
CREATED → IDLE → BIDDING → EXECUTING → REFLECTING
    → PROMOTING → PERSONA_UPDATING → IDLE (循环)
                                   → SPLITTING → IDLE
              IDLE → RETIRING → RETIRED
```

### 能力晋升管道

```
In-context (工作记忆) → Out-of-context (缓冲区)
    → Experience (经验, 置信度升降)
    → Skills (技能, confidence > 0.92)
    → Persona (人物画像, LLM 归纳)
```

### 分裂与退休

- **分裂 (SPLITTING)**: Agent 积累 ≥5 个技能且覆盖 ≥3 个领域时，按主标签拆分为 2-4 个子 Agent，各自继承对应技能
- **退休 (RETIRING)**: 连续 ≥20 次未中标，或 ≥15 次任务后成功率 <20%，触发退休，技能进入全局技能市场

## 关键参数

| 参数 | 值 |
|------|-----|
| 竞标基线权重 | 0.30 |
| 新手探索加成 | +0.20 (经验 <8 条时) |
| 经验初始置信度 | 0.55 ~ 0.75 |
| 技能晋升置信度阈值 | > 0.92 |
| 分裂所需技能数 | ≥ 5 |
| 分裂所需标签多样性 | ≥ 3 |
| 退休新手保护期 | < 15 次任务不检查 |

## 预设 Agent

| Agent | 初始领域 | 预期轨迹 |
|-------|---------|---------|
| Alpha | Web 安全加固 | 积累安全技能，覆盖多领域后分裂 |
| Beta | 性能优化 / 数据库 | 稳定的性能专家 |
| Gamma | 前端开发 | 前端方向持续成长 |
| Delta | 通用 (无预设) | 探索后逐渐找到方向 |

## GUI 操作

- **▶ 开始 / ⏸ 暂停**: 自动运行/暂停模拟
- **⏩ 单步**: 手动推进一个 tick
- **🔄 重置**: 清除所有数据，重新创建 4 个初始 Agent
- **速度滑块**: 调节自动运行间隔 (0.3s ~ 2.0s)

## 界面布局

- **左栏**: Mermaid 状态流转图 (实时高亮当前活跃状态) + 全局指标
- **右栏**: Agent 卡片矩阵，颜色编码表示状态，分裂/退休有独立视觉标识
- **底部**: 滚动事件日志 + 能力晋升管道可视化
