"""
Agent 生命周期 Demo — 状态机核心
每个状态转移都必须通过 before 回调完成数据操作后才能进入下一状态。
"""
from __future__ import annotations
from typing import Callable, Optional
from dataclasses import dataclass, field
from agent_model import Agent, AgentState


# ── 转移定义 ─────────────────────────────────────────────
@dataclass
class Transition:
    trigger: str
    source: AgentState
    dest: AgentState
    condition: Optional[Callable[[Agent], bool]] = None
    before: Optional[Callable[[Agent], None]] = None
    after: Optional[Callable[[Agent], str]] = None  # 返回事件描述


class AgentStateMachine:
    """每个 Agent 持有一个状态机实例"""

    TRANSITION_TABLE: list[Transition] = [
        Transition(
            trigger="init",
            source=AgentState.CREATED,
            dest=AgentState.IDLE,
            before=lambda a: setattr(a, "state", AgentState.IDLE),
            after=lambda a: f"{a.name} 初始化完成，进入待命状态 (IDLE)"
        ),
        Transition(
            trigger="announce_task",
            source=AgentState.IDLE,
            dest=AgentState.BIDDING,
            before=lambda a: setattr(a, "state", AgentState.BIDDING),
            after=lambda a: f"{a.name} 收到任务公告，进入竞标 (BIDDING)"
        ),
        Transition(
            trigger="win_bid",
            source=AgentState.BIDDING,
            dest=AgentState.EXECUTING,
            before=lambda a: setattr(a, "state", AgentState.EXECUTING),
            after=lambda a: f"{a.name} 中标! 进入执行阶段 (EXECUTING)"
        ),
        Transition(
            trigger="lose_bid",
            source=AgentState.BIDDING,
            dest=AgentState.IDLE,
            before=lambda a: setattr(a, "state", AgentState.IDLE),
            after=lambda a: f"{a.name} 未中标，回到待命 (IDLE)"
        ),
        Transition(
            trigger="task_done",
            source=AgentState.EXECUTING,
            dest=AgentState.REFLECTING,
            before=lambda a: setattr(a, "state", AgentState.REFLECTING),
            after=lambda a: f"{a.name} 任务执行完毕，进入反思提取 (REFLECTING)"
        ),
        Transition(
            trigger="reflect_done",
            source=AgentState.REFLECTING,
            dest=AgentState.PROMOTING,
            before=lambda a: setattr(a, "state", AgentState.PROMOTING),
            after=lambda a: f"{a.name} 经验提取完成，检查技能晋升 (PROMOTING)"
        ),
        Transition(
            trigger="promote_done",
            source=AgentState.PROMOTING,
            dest=AgentState.PERSONA_UPDATING,
            condition=lambda a: any(_should_promote(e) for e in a.experiences),
            before=lambda a: setattr(a, "state", AgentState.PERSONA_UPDATING),
            after=lambda a: f"{a.name} 有经验晋升为技能，更新 Persona (PERSONA_UPDATING)"
        ),
        Transition(
            trigger="no_promote",
            source=AgentState.PROMOTING,
            dest=AgentState.IDLE,
            condition=lambda a: not any(_should_promote(e) for e in a.experiences),
            before=lambda a: setattr(a, "state", AgentState.IDLE),
            after=lambda a: f"{a.name} 无经验晋升，回到待命 (IDLE)"
        ),
        Transition(
            trigger="persona_done",
            source=AgentState.PERSONA_UPDATING,
            dest=AgentState.IDLE,
            condition=lambda a: not _should_split(a) and not _should_retire(a),
            before=lambda a: setattr(a, "state", AgentState.IDLE),
            after=lambda a: f"{a.name} Persona 更新完毕，回到待命 (IDLE)"
        ),
        Transition(
            trigger="check_split",
            source=AgentState.PERSONA_UPDATING,
            dest=AgentState.SPLITTING,
            condition=lambda a: _should_split(a),
            before=lambda a: setattr(a, "state", AgentState.SPLITTING),
            after=lambda a: f"{a.name} 技能聚类分离，触发分裂 (SPLITTING)"
        ),
        Transition(
            trigger="split_done",
            source=AgentState.SPLITTING,
            dest=AgentState.IDLE,
            before=lambda a: setattr(a, "state", AgentState.IDLE),
            after=lambda a: f"{a.name} 分裂完成，子 Agent 就绪"
        ),
        Transition(
            trigger="degrade",
            source=AgentState.IDLE,
            dest=AgentState.RETIRING,
            condition=lambda a: _should_retire(a),
            before=lambda a: setattr(a, "state", AgentState.RETIRING),
            after=lambda a: f"{a.name} 满足退休条件，进入退休流程 (RETIRING)"
        ),
        Transition(
            trigger="retire_done",
            source=AgentState.RETIRING,
            dest=AgentState.RETIRED,
            before=lambda a: setattr(a, "state", AgentState.RETIRED),
            after=lambda a: f"{a.name} 退休完成 (RETIRED)"
        ),
    ]

    _transition_map: dict[tuple[AgentState, str], Transition]

    def __init__(self):
        self._transition_map = {}
        for t in self.TRANSITION_TABLE:
            key = (t.source, t.trigger)
            if key in self._transition_map:
                # 有多个同 source+trigger 的转移（带 condition 分支）
                # 存储为 list
                existing = self._transition_map[key]
                if isinstance(existing, list):
                    existing.append(t)
                else:
                    self._transition_map[key] = [existing, t]
            else:
                self._transition_map[key] = t

    def fire(self, agent: Agent, trigger: str) -> Optional[str]:
        """触发状态转移。返回事件描述字符串，如触发失败返回 None。"""
        key = (agent.state, trigger)
        entry = self._transition_map.get(key)
        if entry is None:
            return None

        # 解析可能的多个转移（含 condition 分支）
        candidates = entry if isinstance(entry, list) else [entry]
        matched = None
        for t in candidates:
            if t.condition is None or t.condition(agent):
                matched = t
                break

        if matched is None:
            return None

        # 执行转移
        if matched.before:
            matched.before(agent)
        agent.state = matched.dest
        event = matched.after(agent) if matched.after else f"{agent.name}: {matched.source.value} → {matched.dest.value}"
        return event

    def can_fire(self, agent: Agent, trigger: str) -> bool:
        key = (agent.state, trigger)
        entry = self._transition_map.get(key)
        if entry is None:
            return False
        candidates = entry if isinstance(entry, list) else [entry]
        return any(t.condition is None or t.condition(agent) for t in candidates)

    def available_triggers(self, agent: Agent) -> list[str]:
        """返回当前状态下可用的 trigger 列表"""
        triggers = []
        for t in self.TRANSITION_TABLE:
            if t.source == agent.state:
                if t.condition is None or t.condition(agent):
                    triggers.append(t.trigger)
        return list(dict.fromkeys(triggers))  # 去重保序


# ── 条件判断函数 ─────────────────────────────────────────

def _should_promote(exp) -> bool:
    """检查经验是否达到晋升门槛 (Demo: confidence > 0.92)"""
    return exp.confidence > 0.92 and exp.exp_type == "positive"


def _should_split(agent: Agent) -> bool:
    """检查 Agent 是否应该分裂 — Demo 友好版本
    条件：Skills ≥ 5 且覆盖至少 2 个差异明显的子方向
    """
    if len(agent.skills) < 5:
        return False
    tag_groups = set()
    for s in agent.skills:
        for t in s.tags:
            tag_groups.add(t)
    return len(tag_groups) >= 3


def _should_retire(agent: Agent) -> bool:
    """检查 Agent 是否应该退休 — Demo 友好版本。

    退休是状态机中最不可逆的操作。Demo 中应当让 Agent 有充分时间
    展示成长曲线，只有在长期明显偏离时才触发退休。
    """
    total = agent.metrics.total_tasks
    is_newbie = total < 15

    # ── 新手完全受保护，不做任何退休检查 ──
    # （15 次任务的窗口足够 Agent 积累经验和技能）
    if is_newbie:
        return False

    # ── 连续未中标: 超过 20 次才触发 ──
    if agent.consecutive_missed_bids >= 20:
        return True

    # ── 成功率: ≥15 条记录且 < 20% 才触发 ──
    recent = agent.recent_successes[-10:] if agent.recent_successes else []
    if len(recent) >= 15:
        rate = sum(recent[:15]) / 15
        if rate < 0.20:
            return True

    return False
