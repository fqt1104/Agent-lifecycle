"""
Agent 生命周期 Demo — Mock 数据层
模拟向量检索、LLM 反思、Persona 归纳等。
所有逻辑真实执行，输入数据为模拟值。
"""
from __future__ import annotations
import random
import uuid
import time
import math
from typing import Optional
from agent_model import (
    Agent, Experience, Skill, Persona, Task, BufferEntry, Metrics,
    AgentState,
    _cosine_sim,
)


# ── 领域向量预设 ─────────────────────────────────────────
DOMAIN_CENTERS: dict[str, list[float]] = {
    "web_security":     [0.92, 0.10, 0.05, 0.08, 0.06, 0.04, 0.03, 0.02],
    "infra_security":   [0.10, 0.90, 0.08, 0.05, 0.04, 0.03, 0.02, 0.01],
    "compliance_audit": [0.05, 0.06, 0.93, 0.04, 0.03, 0.08, 0.02, 0.01],
    "performance":      [0.06, 0.04, 0.05, 0.91, 0.07, 0.03, 0.02, 0.01],
    "frontend":         [0.04, 0.03, 0.05, 0.06, 0.92, 0.02, 0.03, 0.01],
    "database":         [0.03, 0.02, 0.04, 0.05, 0.03, 0.91, 0.02, 0.01],
    "general":          [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
}

DOMAIN_NAMES_CN: dict[str, str] = {
    "web_security": "Web 安全加固",
    "infra_security": "基础设施安全",
    "compliance_audit": "合规审计",
    "performance": "性能优化",
    "frontend": "前端开发",
    "database": "数据库管理",
}


def domain_embedding(domain: str, noise: float = 0.15) -> list[float]:
    """生成带噪声的领域向量"""
    center = DOMAIN_CENTERS.get(domain, DOMAIN_CENTERS["general"])
    vec = [c + random.uniform(-noise, noise) for c in center]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


# ── 任务模板 ─────────────────────────────────────────────
TASK_TEMPLATES = [
    # web_security (4条)
    ("修复 order-service 中的 SQL 注入漏洞", ["web_security", "database"], 0.6),
    ("为 user-auth 模块添加 XSS 防护", ["web_security", "frontend"], 0.5),
    ("修复 order-service 中的 SSRF 漏洞", ["web_security"], 0.65),
    ("配置 WAF 规则防护 API 端点", ["web_security", "infra_security"], 0.55),

    # infra_security (5条)
    ("容器化部署的安全加固：网络隔离与权限最小化", ["infra_security"], 0.7),
    ("K8s 集群 RBAC 策略审计与修复", ["infra_security"], 0.75),
    ("日志收集系统的访问控制加固", ["infra_security"], 0.55),
    ("CI/CD 流水线密钥管理安全改造", ["infra_security"], 0.65),
    ("微服务间 mTLS 通信配置与证书轮换", ["infra_security"], 0.6),

    # compliance_audit (4条)
    ("SOC2 合规审计：代码库安全策略全面检查", ["compliance_audit"], 0.8),
    ("GDPR 数据隐私合规：用户数据处理流程审计", ["compliance_audit"], 0.75),
    ("PCI-DSS 支付数据处理合规检查", ["compliance_audit"], 0.7),
    ("ISO 27001 信息安全管理系统差距分析", ["compliance_audit"], 0.7),

    # performance (5条)
    ("数据库查询性能优化：慢查询分析与索引重构", ["performance", "database"], 0.6),
    ("API 响应时间优化：缓存策略与连接池调优", ["performance"], 0.55),
    ("前端首屏加载性能优化：代码分割与懒加载", ["performance", "frontend"], 0.5),
    ("消息队列消费延迟优化：批量处理与并发控制", ["performance"], 0.55),
    ("CDN 缓存策略调优：命中率从 72% 提升到 95%", ["performance"], 0.6),

    # frontend (5条)
    ("React 组件重构：状态管理从 Context 迁移到 Zustand", ["frontend"], 0.5),
    ("前端表单验证库统一：从 Yup 迁移到 Zod", ["frontend"], 0.45),
    ("组件库无障碍改造：ARIA 标签与键盘导航", ["frontend"], 0.5),
    ("CSS-in-JS 到 Tailwind 的渐进式迁移", ["frontend"], 0.5),
    ("前端 E2E 测试覆盖率从 15% 提升到 60%", ["frontend"], 0.55),

    # database (5条)
    ("数据库迁移脚本安全检查：防止锁表与数据丢失", ["database"], 0.55),
    ("分库分表方案设计：按租户 ID 水平拆分", ["database"], 0.7),
    ("读写分离架构下的主从延迟监控与降级策略", ["database"], 0.65),
    ("Redis 集群模式迁移：从哨兵模式到 Cluster", ["database"], 0.6),
    ("时序数据库选型评估：InfluxDB vs TimescaleDB", ["database"], 0.5),
]

USER_RATINGS = [
    ("完全解决", 1.0),
    ("完全解决", 1.0),
    ("完全解决", 1.0),
    ("完全解决", 1.0),
    ("部分解决", 0.6),
    ("部分解决", 0.6),
    ("未解决", 0.0),
]


def generate_task(domain_focus: Optional[list[str]] = None) -> Task:
    """生成一个随机任务"""
    candidates = TASK_TEMPLATES
    if domain_focus:
        candidates = [t for t in TASK_TEMPLATES if any(d in domain_focus for d in t[1])]
        if not candidates:
            candidates = TASK_TEMPLATES

    desc, domains, difficulty = random.choice(candidates)
    # 取第一个领域生成向量
    emb = domain_embedding(domains[0])
    task_id = f"task-{uuid.uuid4().hex[:6]}"
    return Task(
        task_id=task_id,
        description=desc,
        domain_tags=list(domains),
        difficulty=difficulty,
        embedding=emb,
    )


def simulate_user_rating(agent_domain_overlap: float, difficulty: float,
                          agent_experience_count: int = 0) -> tuple[str, float]:
    """模拟用户评分 — Demo 友好版本。

    设计原则: 这是一份演示 Demo，Agent 应当在大多数任务中成功。
    差异化不体现在成败，而体现在经验质量、技能积累、人格分化上。
    """
    # 基础: 55% + 领域匹配度贡献 - 难度惩罚
    base = 0.55 + agent_domain_overlap * 0.30 - difficulty * 0.15
    # 经验加成 (最多 +0.25)
    exp_bonus = min(agent_experience_count * 0.03, 0.25)
    # 成功率上限 92%
    success_prob = min(base + exp_bonus, 0.92)

    r = random.random()

    if r < success_prob:
        return "完全解决", 1.0
    elif r < success_prob + (1.0 - success_prob) * 0.8:
        return "部分解决", 0.6
    else:
        return "未解决", 0.0


# ── LLM 反思模拟 ────────────────────────────────────────

def simulate_extract_experience(
    agent: Agent,
    buffer_entry: BufferEntry,
    task: Task,
) -> list[Experience]:
    """
    模拟离线 LLM 从缓冲区提取经验。
    根据任务结果生成正/负经验，更新已有经验的置信度。
    """
    new_exps = []
    rating = buffer_entry.user_rating

    # 生成决策描述
    decisions_desc = ""
    for d in buffer_entry.decisions:
        decisions_desc += f"  决策点: {d.get('point', '')} → 选择 {d.get('chosen', '')}\n"

    blockers_desc = ""
    for b in buffer_entry.blockers:
        status = "已解决" if b.get('resolved') else "未解决"
        blockers_desc += f"  卡点: {b.get('blocker', '')} → {b.get('resolution', '')} [{status}]\n"

    if rating == "完全解决":
        exp = Experience(
            id=f"exp-{uuid.uuid4().hex[:6]}",
            description=f"{task.description[:30]}... — 成功经验",
            content=(
                f"方案: 针对 {task.description}，采用以下策略:\n"
                f"{decisions_desc}"
                f"结果: 成功，用户评价「{rating}」\n"
                f"关键假设: {', '.join(buffer_entry.assumptions) if buffer_entry.assumptions else '无特殊假设'}"
            ),
            confidence=round(random.uniform(0.55, 0.75), 2),  # 成功经验从较高位起步
            exp_type="positive",
            tags=list(task.domain_tags),
            source_task_id=task.task_id,
            embedding=domain_embedding(task.domain_tags[0], noise=0.1),
        )
        new_exps.append(exp)

    elif rating == "部分解决":
        # 正经验（部分有效）
        exp_pos = Experience(
            id=f"exp-{uuid.uuid4().hex[:6]}",
            description=f"{task.description[:30]}... — 部分成功（需注意适配）",
            content=(
                f"方案: 针对 {task.description}\n"
                f"{decisions_desc}"
                f"结果: 部分解决，用户评价「{rating}」\n"
                f"注意事项: 方案需要针对具体场景适配\n"
                f"假设: {', '.join(buffer_entry.assumptions) if buffer_entry.assumptions else '无'}"
            ),
            confidence=round(random.uniform(0.40, 0.55), 2),
            exp_type="positive",
            tags=list(task.domain_tags),
            source_task_id=task.task_id,
            embedding=domain_embedding(task.domain_tags[0], noise=0.12),
        )
        # 负经验（什么条件下失败了）
        exp_neg = Experience(
            id=f"exp-{uuid.uuid4().hex[:6]}",
            description=f"{task.description[:30]}... — 失败教训",
            content=f"在 {task.description} 中，部分方案未能完全适用。\n{blockers_desc}\n建议: 执行前验证环境兼容性",
            confidence=round(random.uniform(0.4, 0.6), 2),
            exp_type="negative",
            tags=list(task.domain_tags),
            source_task_id=task.task_id,
            embedding=domain_embedding(task.domain_tags[0], noise=0.1),
        )
        exp_pos.linked_negative_exp = [exp_neg.id]
        new_exps.extend([exp_pos, exp_neg])

    else:  # 未解决
        exp_neg = Experience(
            id=f"exp-{uuid.uuid4().hex[:6]}",
            description=f"{task.description[:30]}... — 失败记录",
            content=f"尝试解决 {task.description} 失败。\n{blockers_desc}\n教训: 方案不适用，需重新评估方法",
            confidence=round(random.uniform(0.5, 0.7), 2),
            exp_type="negative",
            tags=list(task.domain_tags),
            source_task_id=task.task_id,
            embedding=domain_embedding(task.domain_tags[0], noise=0.1),
        )
        new_exps.append(exp_neg)

    # 更新被引用的已有经验的置信度
    for old_exp in agent.experiences:
        effectiveness = buffer_entry.effectiveness_map.get(old_exp.id)
        if effectiveness is None:
            continue
        old_exp.referenced_count += 1
        if effectiveness == "fully_effective":
            old_exp.confidence = min(1.0, old_exp.confidence + 0.10)
        elif effectiveness == "partially_effective":
            old_exp.confidence = min(1.0, old_exp.confidence + 0.08)
        elif effectiveness == "ineffective":
            old_exp.confidence = max(0.0, old_exp.confidence - 0.10)

    return new_exps


def simulate_promote_skills(agent: Agent) -> list[Skill]:
    """模拟经验晋升为技能，返回新晋升的 Skill 列表 (Demo: confidence > 0.90)"""
    new_skills = []
    for exp in list(agent.experiences):
        if exp.confidence > 0.92 and exp.exp_type == "positive":
            skill = Skill(
                id=f"skill-{uuid.uuid4().hex[:6]}",
                description=exp.description.replace("成功经验", "技能").replace("部分成功", "技能"),
                content=f"[技能化] {exp.content}\n\n应用步骤:\n1. 分析目标场景\n2. 应用已验证方案\n3. 验证结果",
                tags=list(exp.tags),
                promoted_from=exp.id,
                embedding=list(exp.embedding),
            )
            new_skills.append(skill)
            agent.skills.append(skill)
            # 原始经验删除（晋升后迁移）
            agent.experiences.remove(exp)

    return new_skills


def simulate_update_persona(agent: Agent, all_agents: list[Agent] = None) -> Persona:
    """模拟 LLM 归纳当前 Agent 的经验+技能，生成新 Persona"""
    # 统计标签分布
    tag_counts: dict[str, int] = {}
    for s in agent.skills:
        for t in s.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    for e in agent.experiences:
        for t in e.tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    top_tags = sorted_tags[:3]

    if not top_tags:
        return agent.persona

    domain_names = [DOMAIN_NAMES_CN.get(t, t) for t, _ in top_tags]
    summary = f"专长: {', '.join(domain_names)}"
    skills_overview = f"掌握 {len(agent.skills)} 项技能，覆盖 {', '.join(domain_names)}"
    exp_coverage = f"积累 {len(agent.experiences)} 条经验"
    rate = agent.metrics.success_rate
    recent = f"近期成功率 {rate:.0%}，共 {agent.metrics.total_tasks} 次任务"
    notes = "成长方向稳定" if rate >= 0.6 else "需关注: 成功率偏低"

    old_version = agent.persona.version
    new_persona = Persona(
        version=old_version + 1,
        summary=summary,
        skills_overview=skills_overview,
        experience_coverage=exp_coverage,
        recent_performance=recent,
        notes=notes,
    )
    agent.persona_history.append(agent.persona)
    agent.persona = new_persona
    return new_persona


def simulate_split(agent: Agent) -> list[Agent]:
    """模拟 Agent 分裂：按技能主标签聚类，每个技能只归一个子 Agent。

    问题修复：之前每个技能按所有标签被分配到多个组，导致同一技能
    出现在多个子 Agent 中，且"other"组膨胀。现在只用主标签（第一个），
    小组合并到最近的大组，上限 4 个子 Agent。
    """
    if not agent.skills:
        return []

    # 按主标签分组（每技能只属于一个组）
    primary_tag_skills: dict[str, list[Skill]] = {}
    for s in agent.skills:
        primary = s.tags[0] if s.tags else "general"
        primary_tag_skills.setdefault(primary, []).append(s)

    # 按组大小排序
    groups = sorted(primary_tag_skills.items(), key=lambda x: len(x[1]), reverse=True)

    # 小组合并：不足 2 个技能的组，合并到最大的相邻组
    merged: list[tuple[str, list[Skill]]] = []
    orphans: list[Skill] = []
    for tag, skills in groups:
        if len(skills) >= 2:
            merged.append((tag, list(skills)))
        else:
            orphans.extend(skills)

    # 孤儿技能分配给最大的组
    if orphans and merged:
        merged[0] = (merged[0][0], merged[0][1] + orphans)
    elif orphans:
        merged.append(("general", orphans))

    if len(merged) < 2:
        return []

    # 上限 4 个子 Agent（取最大的 4 组）
    merged = merged[:4]

    children = []
    for tag, skills in merged:
        skills = list(skills)
        tag_cn = DOMAIN_NAMES_CN.get(tag, tag)
        child_name = f"{agent.name}-{tag_cn[:8]}"
        child = Agent(
            role_id=f"agent-{uuid.uuid4().hex[:6]}",
            name=child_name,
            skills=skills,
            persona=Persona(
                summary=f"继承自 {agent.name}，专攻 {tag_cn}",
                skills_overview=f"继承 {len(skills)} 项技能",
            ),
            tags=[tag],
            domain_embedding=domain_embedding(tag),
            parent_agent=agent.role_id,
        )
        # 分配相关经验
        for exp in agent.experiences:
            if tag in exp.tags:
                child.experiences.append(exp)
        child.state = AgentState.CREATED
        children.append(child)

    return children


def simulate_retire(agent: Agent) -> list[Skill]:
    """
    模拟 Agent 退休：Skills 进入技能市场，Experience 丢弃。
    返回进入市场的 Skills 列表。
    """
    retired_skills = list(agent.skills)
    agent.skills.clear()
    agent.experiences.clear()
    agent.retired_at = time.time()
    agent.persona_history.append(agent.persona)
    return retired_skills
