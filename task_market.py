"""
Agent 生命周期 Demo — 任务市场
Boss 发布任务，Agent 基于 Persona+Skills+Experience 计算竞争力，加权抽签认领。
"""
from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from agent_model import Agent, Task, _cosine_sim
from mock_data import generate_task


@dataclass
class TaskMarket:
    """全局任务市场单例"""
    task_queue: list[Task] = field(default_factory=list)
    completed_tasks: list[Task] = field(default_factory=list)
    task_history: list[dict] = field(default_factory=list)

    def generate_tasks(self, count: int = 3):
        """生成新任务加入队列"""
        for _ in range(count):
            self.task_queue.append(generate_task())

    def announce_task(self) -> Task | None:
        """发布队首任务，如队列为空返回 None"""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None

    def compute_bid_weight(self, agent: Agent, task: Task) -> float:
        """
        计算 Agent 对任务的竞争力权重。
        考虑因素:
        1. 领域匹配度（经验向量与任务向量的余弦相似度）
        2. 技能覆盖度（技能 tag 与任务 tag 的 Jaccard）
        3. 历史成功率
        4. 基线权重（白板 Agent 最低保障）
        """
        BASELINE = 0.30

        # 经验匹配度
        exp_similarities = []
        for exp in agent.experiences:
            sim = _cosine_sim(exp.embedding, task.embedding)
            exp_similarities.append(sim * exp.confidence)

        if exp_similarities:
            exp_match = sum(exp_similarities) / len(exp_similarities)
        else:
            exp_match = 0.0

        # 技能覆盖度
        if agent.skills and task.domain_tags:
            skill_tags = set()
            for s in agent.skills:
                skill_tags.update(s.tags)
            task_tags = set(task.domain_tags)
            jaccard = len(skill_tags & task_tags) / max(len(skill_tags | task_tags), 1)
        else:
            jaccard = 0.0

        # 成功率加成
        success_bonus = agent.metrics.success_rate * 0.2

        # 探索加成：新手和经验沉淀不足的 Agent 获得显著加权
        # 保证每个 Agent 都有公平的成长窗口
        exploration_bonus = 0.0
        if len(agent.experiences) < 8:
            exploration_bonus = 0.20  # 新手大幅保护
        elif agent.consecutive_missed_bids >= 3:
            exploration_bonus = min(0.03 * agent.consecutive_missed_bids, 0.18)

        # Persona 与任务的语义匹配
        persona_sim = _cosine_sim(agent.domain_embedding, task.embedding) if any(v != 0 for v in agent.domain_embedding) else 0.0

        weight = BASELINE + exp_match * 0.4 + jaccard * 0.25 + success_bonus + persona_sim * 0.15 + exploration_bonus
        return max(BASELINE, min(weight, 1.0))

    def weighted_lottery(self, agents: list[Agent], task: Task) -> Agent | None:
        """
        加权抽签：权重越高的 Agent 中标概率越大，但非确定性。
        使用随机 acceptance 而非排序，体现"竞争力"而非"排名"。
        """
        if not agents:
            return None

        weights = []
        for a in agents:
            w = self.compute_bid_weight(a, task)
            a.bid_weight = w
            weights.append(w)

        total = sum(weights)
        if total == 0:
            return random.choice(agents)

        # 加权随机选择
        r = random.random() * total
        cumulative = 0.0
        for a, w in zip(agents, weights):
            cumulative += w
            if r <= cumulative:
                return a

        return agents[-1]  # fallback

    @property
    def queue_size(self) -> int:
        return len(self.task_queue)

    @property
    def total_completed(self) -> int:
        return len(self.completed_tasks)
