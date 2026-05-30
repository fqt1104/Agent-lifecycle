"""
Agent 生命周期 Demo — 数据模型
基于 方向B：Agent 角色与记忆系统 RFC
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid
import time


# ── 状态枚举 ─────────────────────────────────────────────
class AgentState(str, Enum):
    CREATED = "CREATED"
    IDLE = "IDLE"
    BIDDING = "BIDDING"
    EXECUTING = "EXECUTING"
    REFLECTING = "REFLECTING"
    PROMOTING = "PROMOTING"
    PERSONA_UPDATING = "PERSONA_UPDATING"
    SPLITTING = "SPLITTING"
    RETIRING = "RETIRING"
    RETIRED = "RETIRED"


# ── Persona ──────────────────────────────────────────────
@dataclass
class Persona:
    version: int = 1
    summary: str = "白板 Agent，尚无专长领域"
    skills_overview: str = "暂无技能"
    experience_coverage: str = "暂无经验"
    recent_performance: str = "尚未执行任务"
    notes: str = ""
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "summary": self.summary,
            "skills_overview": self.skills_overview,
            "experience_coverage": self.experience_coverage,
            "recent_performance": self.recent_performance,
            "notes": self.notes,
        }


# ── Experience ───────────────────────────────────────────
@dataclass
class Experience:
    id: str
    description: str            # ≤3句，检索入口
    content: str                # 完整方案 + 决策 + 结果
    confidence: float           # 0.0 ~ 1.0
    exp_type: str               # "positive" | "negative"
    tags: list[str] = field(default_factory=list)
    linked_negative_exp: list[str] = field(default_factory=list)
    source_task_id: str = ""
    referenced_count: int = 0
    embedding: list[float] = field(default_factory=lambda: [0.0] * 8)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "content": self.content,
            "confidence": round(self.confidence, 2),
            "type": self.exp_type,
            "tags": self.tags,
            "referenced_count": self.referenced_count,
        }


# ── Skill ────────────────────────────────────────────────
@dataclass
class Skill:
    id: str
    description: str
    content: str                # 步骤 + 参数 + 注意事项
    version: str = "1.0.0"
    review_status: str = "approved"
    tags: list[str] = field(default_factory=list)
    promoted_from: str = ""     # 来源 experience_id
    promoted_at: float = field(default_factory=time.time)
    embedding: list[float] = field(default_factory=lambda: [0.0] * 8)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "version": self.version,
            "tags": self.tags,
            "promoted_from": self.promoted_from,
        }


# ── Metrics ──────────────────────────────────────────────
@dataclass
class Metrics:
    total_tasks: int = 0
    success_count: int = 0
    fail_count: int = 0
    skill_count: int = 0
    experience_count: int = 0
    token_cost_total: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return self.success_count / self.total_tasks


# ── Buffer 条目 ──────────────────────────────────────────
@dataclass
class BufferEntry:
    task_id: str
    task_description: str
    user_rating: str            # "完全解决" | "部分解决" | "未解决"
    decisions: list[dict] = field(default_factory=list)
    blockers: list[dict] = field(default_factory=list)
    referenced_exp_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    effectiveness_map: dict = field(default_factory=dict)  # exp_id → effectiveness


# ── Task ─────────────────────────────────────────────────
@dataclass
class Task:
    task_id: str
    description: str
    domain_tags: list[str]
    difficulty: float          # 0.0 ~ 1.0
    embedding: list[float] = field(default_factory=lambda: [0.0] * 8)
    result_rating: Optional[str] = None  # 模拟用户评分


# ── Agent ────────────────────────────────────────────────
@dataclass
class Agent:
    role_id: str
    name: str
    persona: Persona = field(default_factory=Persona)
    experiences: list[Experience] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    state: AgentState = AgentState.CREATED
    buffer: list[BufferEntry] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    domain_embedding: list[float] = field(default_factory=lambda: [0.0] * 8)
    initial_domain_embedding: list[float] = field(default_factory=lambda: [0.0] * 8)

    # 运行时状态
    bid_weight: float = 0.0
    consecutive_missed_bids: int = 0
    current_task: Optional[Task] = None
    persona_history: list[Persona] = field(default_factory=list)
    retired_at: Optional[float] = None
    parent_agent: str = ""      # 分裂来源

    # 分裂/退休追踪
    total_task_windows: int = 0
    recent_successes: list[bool] = field(default_factory=list)  # 最近10次

    def __post_init__(self):
        if not self.role_id:
            self.role_id = f"agent-{uuid.uuid4().hex[:8]}"
        # 保存初始领域向量，防止后续经验漂移抹平个性
        if all(v == 0.0 for v in self.initial_domain_embedding) and any(v != 0.0 for v in self.domain_embedding):
            self.initial_domain_embedding = list(self.domain_embedding)

    def to_card(self) -> dict:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "state": self.state.value,
            "persona_summary": self.persona.summary,
            "skill_count": len(self.skills),
            "experience_count": len(self.experiences),
            "avg_confidence": round(
                sum(e.confidence for e in self.experiences) / max(len(self.experiences), 1), 2
            ),
            "success_rate": round(self.metrics.success_rate * 100, 1),
            "total_tasks": self.metrics.total_tasks,
            "bid_weight": round(self.bid_weight, 3),
            "tags": self.tags,
        }


# ── 技能市场（全局单例） ─────────────────────────────────
@dataclass
class SkillMarket:
    skills: list[Skill] = field(default_factory=list)

    def add(self, skill: Skill):
        self.skills.append(skill)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Skill]:
        scored = [(s, _cosine_sim(query_embedding, s.embedding)) for s in self.skills]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]


# ── 工具函数 ─────────────────────────────────────────────
def _cosine_sim(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
