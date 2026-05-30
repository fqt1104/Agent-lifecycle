"""
Agent 生命周期 Demo — 模拟引擎
协调所有 Agent 的状态流转、任务分配、经验提取、分裂/退休。
每个 tick 推进一轮完整生命周期。
"""
from __future__ import annotations
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from agent_model import Agent, AgentState, Skill, SkillMarket, BufferEntry, _cosine_sim
from state_machine import AgentStateMachine, _should_retire
from task_market import TaskMarket
from mock_data import (
    simulate_extract_experience,
    simulate_promote_skills,
    simulate_update_persona,
    simulate_split,
    simulate_retire,
    simulate_user_rating,
    domain_embedding,
    DOMAIN_NAMES_CN,
    generate_task,
)


@dataclass
class SimEngine:
    """模拟引擎：每个 tick 推进所有 Agent 一个生命周期轮次"""
    agents: list[Agent] = field(default_factory=list)
    market: TaskMarket = field(default_factory=TaskMarket)
    skill_market: SkillMarket = field(default_factory=SkillMarket)
    tick_count: int = 0
    event_log: list[dict] = field(default_factory=list)
    running: bool = False
    speed: float = 1.0  # 每个 tick 的间隔秒数（UI 控制）
    _domain_rotation: list[str] = field(default_factory=lambda: [
        "web_security", "performance", "frontend", "database",
        "infra_security", "compliance_audit",
    ])
    _rotation_idx: int = 0

    def add_agent(self, name: str, tags: list[str] = None, domain: str = "general") -> Agent:
        """创建并添加一个白板 Agent"""
        agent = Agent(
            role_id=f"agent-{uuid.uuid4().hex[:6]}",
            name=name,
            tags=tags or [],
            domain_embedding=domain_embedding(domain),
        )
        sm = AgentStateMachine()
        agent.state_machine = sm
        sm.fire(agent, "init")
        self.agents.append(agent)
        self._log("system", f"新 Agent {agent.name} 创建，进入待命 (IDLE)")
        return agent

    def _log(self, source: str, message: str, level: str = "info"):
        self.event_log.append({
            "tick": self.tick_count,
            "time": time.time(),
            "source": source,
            "message": message,
            "level": level,
        })

    def tick(self) -> list[dict]:
        """
        执行一个模拟 tick:
        1. Boss 发布新任务
        2. 所有 IDLE Agent 进入 BIDDING
        3. 加权抽签决定中标者
        4. 中标者执行任务 → 反思 → 晋升 → Persona 更新
        5. 检查分裂/退休条件
        """
        self.tick_count += 1
        new_events = []

        # ── 1. Boss 发布任务 ──
        # 按领域轮转生成任务，避免单一领域垄断
        if self.market.queue_size == 0:
            # 同时生成 3 个任务：当前轮转领域、随机领域、活跃 Agent 最多的领域
            rot_domain = self._domain_rotation[self._rotation_idx % len(self._domain_rotation)]
            self._rotation_idx += 1

            # 找出当前活跃 Agent 最多的领域
            agent_domains = set()
            for a in self.get_active_agents():
                for t in a.tags:
                    agent_domains.add(t)

            domains_to_use = [rot_domain]
            if agent_domains:
                domains_to_use.append(random.choice(list(agent_domains)))
            domains_to_use.append(random.choice(list(DOMAIN_NAMES_CN.keys())))

            for dom in domains_to_use:
                self.market.task_queue.append(generate_task(domain_focus=[dom]))

        task = self.market.announce_task()
        if task is None:
            return new_events

        self._log("Boss", f"发布任务 #{self.tick_count}: 「{task.description}」(领域: {', '.join(task.domain_tags)})")

        # ── 2. 将所有 IDLE Agent 置为 BIDDING ──
        active_agents = [a for a in self.agents if a.state == AgentState.IDLE]
        for a in active_agents:
            sm = getattr(a, 'state_machine', AgentStateMachine())
            event = sm.fire(a, "announce_task")
            if event:
                self._log(a.name, event)

        # 等待退休检查的 agents 在这次 tick 也可能被触发
        for a in list(self.agents):
            if a.state == AgentState.IDLE and _should_retire(a):
                sm = getattr(a, 'state_machine', AgentStateMachine())
                event = sm.fire(a, "degrade")
                if event:
                    self._log(a.name, event, "warning")

        # ── 3. 加权抽签 ──
        bidding_agents = [a for a in self.agents if a.state == AgentState.BIDDING]
        for a in bidding_agents:
            self.market.compute_bid_weight(a, task)

        winner = self.market.weighted_lottery(bidding_agents, task)

        for a in bidding_agents:
            sm = getattr(a, 'state_machine', AgentStateMachine())
            if a == winner:
                a.current_task = task
                event = sm.fire(a, "win_bid")
                if event:
                    self._log(a.name, event, "success")
                a.consecutive_missed_bids = 0
            else:
                event = sm.fire(a, "lose_bid")
                if event:
                    self._log(a.name, event)
                a.consecutive_missed_bids += 1
                a.total_task_windows += 1

        if winner is None:
            return new_events

        # ── 4. 中标者执行：EXECUTING → REFLECTING → PROMOTING → PERSONA_UPDATING ──
        agent = winner
        sm = getattr(agent, 'state_machine', AgentStateMachine())

        # 4a. 模拟 Driver 执行
        domain_overlap = _cosine_sim(agent.domain_embedding, task.embedding)
        rating, _ = simulate_user_rating(domain_overlap, task.difficulty,
                                          agent_experience_count=len(agent.experiences))
        task.result_rating = rating

        # 模拟 Driver 返回六字段
        buffer_entry = BufferEntry(
            task_id=task.task_id,
            task_description=task.description,
            user_rating=rating,
            decisions=[
                {
                    "point": f"选择 {task.description[:20]} 的技术方案",
                    "options": ["方案A: 适配型改造", "方案B: 全新重构"],
                    "chosen": "方案A: 适配型改造",
                    "reason": "降低风险，利用现有基础"
                }
            ],
            blockers=[],
            assumptions=["目标环境与参考环境兼容"],
        )

        # 模拟引用的经验效果反馈
        related_exps = [e for e in agent.experiences if any(t in e.tags for t in task.domain_tags)][:3]
        for e in related_exps:
            eff = random.choice(["fully_effective", "partially_effective", "partially_effective", "ineffective"])
            buffer_entry.referenced_exp_ids.append(e.id)
            buffer_entry.effectiveness_map[e.id] = eff

        agent.buffer.append(buffer_entry)
        self._log(agent.name, f"Driver 执行完毕，用户评价「{rating}」")

        # 更新 Metrics
        agent.metrics.total_tasks += 1
        is_success = rating == "完全解决"
        if is_success:
            agent.metrics.success_count += 1
        else:
            agent.metrics.fail_count += 1
        agent.recent_successes.append(is_success)
        if len(agent.recent_successes) > 10:
            agent.recent_successes = agent.recent_successes[-10:]

        # 任务完成 → REFLECTING
        event = sm.fire(agent, "task_done")
        if event:
            self._log(agent.name, event)

        # 4b. 反思提取经验
        new_exps = simulate_extract_experience(agent, buffer_entry, task)
        for exp in new_exps:
            agent.experiences.append(exp)
        agent.metrics.experience_count = len(agent.experiences)
        self._log(agent.name, f"离线 LLM 反思: 提取 {len(new_exps)} 条经验 (当前共 {len(agent.experiences)} 条)")

        # REFLECTING → PROMOTING
        event = sm.fire(agent, "reflect_done")
        if event:
            self._log(agent.name, event)

        # 4c. 检查经验晋升 — 注意顺序!
        # 必须先检查条件再做状态转移，最后才执行晋升（晋升会删除经验）
        # 否则 promote_done 的 condition 检查经验时发现已被删除，转移失败
        has_promotable = any(
            e.confidence > 0.92 and e.exp_type == "positive"
            for e in agent.experiences
        )

        if has_promotable:
            event = sm.fire(agent, "promote_done")
        else:
            event = sm.fire(agent, "no_promote")

        if event:
            self._log(agent.name, event)

        # 状态转移完成后，再执行实际的技能晋升
        new_skills = simulate_promote_skills(agent)
        agent.metrics.skill_count = len(agent.skills)
        for sk in new_skills:
            self._log(agent.name, f"⭐ 经验 {sk.promoted_from} 晋升为技能: {sk.description}", "success")
            self.skill_market.add(sk)

        # 如果进入了 PERSONA_UPDATING
        if agent.state == AgentState.PERSONA_UPDATING:
            # 4d. 更新 Persona
            new_persona = simulate_update_persona(agent, self.agents)
            self._log(agent.name, f"Persona 更新: {new_persona.summary} (v{new_persona.version})")

            # ── 5. 检查分裂条件 ──
            if len(agent.skills) >= 5:
                tag_groups = set()
                for s in agent.skills:
                    for t in s.tags:
                        tag_groups.add(t)
                if len(tag_groups) >= 3:
                    event = sm.fire(agent, "check_split")
                    if event:
                        self._log(agent.name, f"🔀 技能聚类分离: {len(tag_groups)} 个子方向，触发分裂!", "warning")
                    # 执行分裂
                    children = simulate_split(agent)
                    if children:
                        for child in children:
                            child_sm = AgentStateMachine()
                            child.state_machine = child_sm
                            child_sm.fire(child, "init")
                            self.agents.append(child)
                            self._log("system", f"  新 Agent {child.name} 诞生: {child.persona.summary}")
                    event2 = sm.fire(agent, "split_done")
                    if event2:
                        self._log(agent.name, event2)
                    # 原 Agent 技能已继承给子Agent，清空后不再重复触发分裂
                    agent.skills.clear()
                    agent.metrics.skill_count = 0
                    agent.experiences.clear()
                    agent.metrics.experience_count = 0
                    agent.state = AgentState.IDLE
                else:
                    event = sm.fire(agent, "persona_done")
                    if event:
                        self._log(agent.name, event)
            else:
                event = sm.fire(agent, "persona_done")
                if event:
                    self._log(agent.name, event)

        # 6. 检查退休
        for a in list(self.agents):
            if a.state == AgentState.IDLE and _should_retire(a):
                sm_a = getattr(a, 'state_machine', AgentStateMachine())
                event = sm_a.fire(a, "degrade")
                if event:
                    self._log(a.name, event, "warning")
                # 执行退休
                skills_to_market = simulate_retire(a)
                for sk in skills_to_market:
                    self.skill_market.add(sk)
                event2 = sm_a.fire(a, "retire_done")
                if event2:
                    self._log(a.name, f"👋 退休: {len(skills_to_market)} 项技能进入市场，经验已清理", "warning")

        # 更新 domain embedding — 慢速混合而非完全替换
        # 70% 保持初始方向 (Agent 的"先天特长")，30% 反映经验积累
        # 这样每个 Agent 保持清晰的领域辨识度，不会都漂向同一方向
        if agent.experiences:
            avg_emb = [0.0] * 8
            for exp in agent.experiences:
                for i in range(8):
                    avg_emb[i] += exp.embedding[i]
            n = len(agent.experiences)
            exp_avg = [v / n for v in avg_emb]
            # 70% 初始 + 30% 经验均值
            init = agent.initial_domain_embedding if any(v != 0.0 for v in agent.initial_domain_embedding) else agent.domain_embedding
            agent.domain_embedding = [
                0.7 * init[i] + 0.3 * exp_avg[i]
                for i in range(8)
            ]

        self.market.completed_tasks.append(task)
        return self.event_log[-5:]  # 返回最近事件

    def tick_retirement_check(self):
        """独立检查所有 Agent 的退休条件"""
        for a in list(self.agents):
            if a.state == AgentState.IDLE and _should_retire(a):
                sm = getattr(a, 'state_machine', AgentStateMachine())
                event = sm.fire(a, "degrade")
                if event:
                    self._log(a.name, event, "warning")
                skills_to_market = simulate_retire(a)
                for sk in skills_to_market:
                    self.skill_market.add(sk)
                event2 = sm.fire(a, "retire_done")
                if event2:
                    self._log(a.name, f"👋 退休: {len(skills_to_market)} 项技能进入市场", "warning")

    def get_active_agents(self) -> list[Agent]:
        """获取非终态的 Agent"""
        return [a for a in self.agents if a.state != AgentState.RETIRED]

    def get_state_summary(self) -> dict:
        """获取全局状态摘要"""
        state_counts = {}
        for a in self.agents:
            s = a.state.value
            state_counts[s] = state_counts.get(s, 0) + 1
        return {
            "tick": self.tick_count,
            "total_agents": len(self.agents),
            "active_agents": len(self.get_active_agents()),
            "market_skills": len(self.skill_market.skills),
            "tasks_completed": self.market.total_completed,
            "tasks_queued": self.market.queue_size,
            "state_distribution": state_counts,
        }
