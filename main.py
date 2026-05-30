"""
Agent 生命周期演示 Demo — Streamlit UI
基于 方向B：Agent 角色与记忆系统 RFC
"""
import streamlit as st
import time
from datetime import datetime
from agent_model import AgentState
from sim_engine import SimEngine

# ── 页面配置 ──
st.set_page_config(
    page_title="Agent 生命周期演示",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 自定义 CSS ──
st.markdown("""
<style>
    .agent-card {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        background: white;
        transition: all 0.3s ease;
    }
    .agent-card.CREATED { border-color: #90caf9; }
    .agent-card.IDLE { border-color: #a5d6a7; }
    .agent-card.BIDDING { border-color: #ffcc80; }
    .agent-card.EXECUTING { border-color: #ef9a9a; }
    .agent-card.REFLECTING { border-color: #ce93d8; }
    .agent-card.PROMOTING { border-color: #80deea; }
    .agent-card.PERSONA_UPDATING { border-color: #f48fb1; }
    .agent-card.SPLITTING { border-color: #ffab91; }
    .agent-card.RETIRING { border-color: #bdbdbd; }
    .agent-card.RETIRED { border-color: #9e9e9e; opacity: 0.6; }
    .agent-card.SPLIT_PARENT { border-color: #bdbdbd; background: #f5f5f5; opacity: 0.7; }

    .state-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    .state-badge.CREATED { background: #42a5f5; }
    .state-badge.IDLE { background: #66bb6a; }
    .state-badge.BIDDING { background: #ffa726; }
    .state-badge.EXECUTING { background: #ef5350; }
    .state-badge.REFLECTING { background: #ab47bc; }
    .state-badge.PROMOTING { background: #26c6da; }
    .state-badge.PERSONA_UPDATING { background: #ec407a; }
    .state-badge.SPLITTING { background: #ff7043; }
    .state-badge.RETIRING { background: #78909c; }
    .state-badge.RETIRED { background: #9e9e9e; }

    .log-entry {
        padding: 6px 12px;
        margin: 2px 0;
        border-left: 3px solid #ccc;
        font-size: 13px;
        font-family: 'Consolas', monospace;
    }
    .log-entry.info { border-left-color: #42a5f5; }
    .log-entry.success { border-left-color: #66bb6a; background: #f1f8e9; }
    .log-entry.warning { border-left-color: #ffa726; background: #fff8e1; }

    .metric-value { font-size: 24px; font-weight: bold; color: #1565c0; }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; }

    @keyframes highlight {
        0% { background: #fff9c4; }
        100% { background: transparent; }
    }
    .highlight { animation: highlight 1.5s ease-out; }
</style>
""", unsafe_allow_html=True)

# ── 初始化 Session State ──
if "engine" not in st.session_state:
    st.session_state.engine = SimEngine()
    # 创建预设 Agent
    engine = st.session_state.engine
    engine.add_agent("Alpha", tags=["web_security"], domain="web_security")
    engine.add_agent("Beta", tags=["performance", "database"], domain="performance")
    engine.add_agent("Gamma", tags=["frontend"], domain="frontend")
    engine.add_agent("Delta", tags=["general"], domain="general")

if "auto_running" not in st.session_state:
    st.session_state.auto_running = False

if "tick_speed" not in st.session_state:
    st.session_state.tick_speed = 0.8

engine = st.session_state.engine


# ── 状态机 Mermaid 图生成 ──
def generate_mermaid(engine) -> str:
    """生成带当前状态高亮的 Mermaid 状态图"""
    agent_states = set()
    for a in engine.agents:
        agent_states.add(a.state.value)

    # 标记哪些状态有 Agent 当前在其中
    def _highlight(state_name):
        if state_name in agent_states:
            return f'{state_name}:::active'
        return state_name

    return f"""
stateDiagram-v2
    [*] --> CREATED
    CREATED --> IDLE : init
    IDLE --> BIDDING : announce
    BIDDING --> EXECUTING : win_bid
    BIDDING --> IDLE : lose_bid
    EXECUTING --> REFLECTING : task_done
    REFLECTING --> PROMOTING : reflect_done
    PROMOTING --> PERSONA_UPDATING : promote_done
    PROMOTING --> IDLE : no_promote
    PERSONA_UPDATING --> IDLE : persona_done
    PERSONA_UPDATING --> SPLITTING : check_split
    SPLITTING --> IDLE : split_done
    IDLE --> RETIRING : degrade
    RETIRING --> RETIRED : retire_done
    RETIRED --> [*]

    classDef active fill:#ffcdd2,stroke:#ef5350,stroke-width:2px
    class {','.join(s for s in agent_states)} active
"""


# ── 渲染 Agent 卡片 ──
def render_agent_card(agent, all_agents=None):
    card = agent.to_card()
    state_val = card["state"]
    state_class = state_val

    # 检查此 Agent 是否已分裂（有子 Agent 指向它）
    is_split_parent = False
    children_names = []
    if all_agents:
        children = [a for a in all_agents if a.parent_agent == agent.role_id]
        if children:
            is_split_parent = True
            children_names = [a.name for a in children]

    # 已分裂的父 Agent 使用灰色主题
    card_class = state_class
    badge_class = state_class
    if is_split_parent:
        card_class = "SPLIT_PARENT"
        badge_class = "RETIRED"  # 灰色徽章

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            name_display = f"### {card['name']}" if not is_split_parent else f"### ~~{card['name']}~~"
            st.markdown(name_display)
            badges = f'<span class="state-badge {badge_class}">{state_val}</span>'
            if is_split_parent:
                badges += (
                    f' <span style="display:inline-block; padding:4px 10px; border-radius:12px; '
                    f'font-size:11px; background:#e0e0e0; color:#757575;">'
                    f'🔀 已分裂 → {", ".join(children_names)}</span>'
                )
            st.markdown(badges, unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-value">{card["avg_confidence"]:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">均置信度</div>', unsafe_allow_html=True)

        # 指标行
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("技能", card["skill_count"])
        with m2:
            st.metric("经验", card["experience_count"])
        with m3:
            st.metric("成功率", f"{card['success_rate']:.0f}%")
        with m4:
            st.metric("任务数", card["total_tasks"])

        # Persona 摘要
        if card["persona_summary"] and card["persona_summary"] != "白板 Agent，尚无专长领域":
            st.caption(f"🎯 {card['persona_summary']}")

        # 技能列表
        if agent.skills:
            skill_tags = set()
            for s in agent.skills:
                for t in s.tags:
                    skill_tags.add(t)
            tags_str = " | ".join(f"`{t}`" for t in list(skill_tags)[:5])
            st.caption(f"📦 技能领域: {tags_str}")


# ── 主界面 ──
def main():
    st.title("🎯 Agent 生命周期演示")
    st.caption("基于 方向B：Agent 角色与记忆系统 RFC — Agent 不是被预设定义的，而是在任务市场竞争中逐渐成长为有辨识度的角色")

    # ── 第一行：控制面板 + 全局指标 ──
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns([1, 1, 1, 1, 1, 1])

    with ctrl_col1:
        if st.button("▶ 开始", use_container_width=True, disabled=st.session_state.auto_running):
            st.session_state.auto_running = True

    with ctrl_col2:
        if st.button("⏸ 暂停", use_container_width=True, disabled=not st.session_state.auto_running):
            st.session_state.auto_running = False

    with ctrl_col3:
        if st.button("⏩ 单步", use_container_width=True):
            engine.tick()
            st.rerun()

    with ctrl_col4:
        if st.button("🔄 重置", use_container_width=True):
            st.session_state.engine = SimEngine()
            e = st.session_state.engine
            e.add_agent("Alpha", tags=["web_security"], domain="web_security")
            e.add_agent("Beta", tags=["performance", "database"], domain="performance")
            e.add_agent("Gamma", tags=["frontend"], domain="frontend")
            e.add_agent("Delta", tags=["general"], domain="general")
            st.session_state.auto_running = False
            st.rerun()

    with ctrl_col5:
        speed = st.select_slider(
            "速度",
            options=[0.3, 0.5, 0.8, 1.2, 2.0],
            value=st.session_state.tick_speed,
            format_func=lambda x: f"{x:.1f}s",
        )
        st.session_state.tick_speed = speed

    with ctrl_col6:
        summary = engine.get_state_summary()
        st.metric("Tick", engine.tick_count)

    # ── 第二行：状态机图 + Agent 面板 ──
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("📊 状态流转图")
        mermaid_code = generate_mermaid(engine)
        st.markdown(f"```mermaid\n{mermaid_code}\n```")

        # 全局指标
        st.divider()
        s = engine.get_state_summary()
        st.metric("活跃 Agent", f"{s['active_agents']}/{s['total_agents']}")
        st.metric("技能市场", f"{s['market_skills']} 项")
        st.metric("已完成任务", s['tasks_completed'])
        st.metric("待处理任务", s['tasks_queued'])

        # 状态分布
        st.caption("状态分布:")
        for state_name, count in s['state_distribution'].items():
            st.caption(f"  {state_name}: {count}")

    with right_col:
        st.subheader("🤖 Agent 面板")

        # 划分活跃和已退休
        active = [a for a in engine.agents if a.state != AgentState.RETIRED]
        retired = [a for a in engine.agents if a.state == AgentState.RETIRED]

        # 活跃 Agent 网格 (每行 2 个)
        if active:
            for i in range(0, len(active), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(active):
                        with cols[j]:
                            cls = f"agent-card {active[i+j].state.value}"
                            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
                            render_agent_card(active[i + j], engine.agents)
                            st.markdown('</div>', unsafe_allow_html=True)

        # 已分裂的 Agent（有子Agent的父Agent）
        split_parents = [a for a in engine.agents if a.parent_agent == "" and any(c.parent_agent == a.role_id for c in engine.agents)]
        if split_parents:
            st.divider()
            for sp in split_parents:
                children = [c for c in engine.agents if c.parent_agent == sp.role_id]
                st.markdown(
                    f'<div style="background:#e8f5e9; border-left:4px solid #66bb6a; '
                    f'padding:8px 12px; margin:4px 0; border-radius:4px;">'
                    f'🔀 <b>{sp.name}</b> 已分裂为 {len(children)} 个子 Agent: '
                    f'{", ".join(c.name for c in children)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # 退休 Agent — 独立小卡片
        if retired:
            st.divider()
            for a in retired:
                skills_kept = sum(1 for s in engine.skill_market.skills)  # 近似值
                st.markdown(f'''
                <div style="
                    background: #fafafa;
                    border: 2px dashed #bdbdbd;
                    border-radius: 10px;
                    padding: 12px 16px;
                    margin: 6px 0;
                    opacity: 0.75;
                ">
                    <span style="font-size:16px; font-weight:bold; text-decoration:line-through; color:#888;">
                        👋 {a.name}
                    </span>
                    <span style="
                        display:inline-block; margin-left:10px; padding:2px 10px;
                        border-radius:12px; font-size:11px;
                        background:#9e9e9e; color:white;
                    ">RETIRED</span>
                    <div style="margin-top:4px; font-size:12px; color:#999;">
                        原 Persona: {a.persona.summary if a.persona_history else a.persona.summary}
                        &nbsp;|&nbsp;
                        技能已进入市场
                        &nbsp;|&nbsp;
                        经验已清理
                    </div>
                </div>
                ''', unsafe_allow_html=True)

    # ── 第三行：事件日志 ──
    st.divider()
    st.subheader("📜 事件日志")

    log_container = st.container()
    # 用 CSS 限制日志区域高度，保证跨版本兼容
    st.markdown('<div style="max-height: 350px; overflow-y: auto; padding: 8px; border: 1px solid #e0e0e0; border-radius: 8px;">', unsafe_allow_html=True)
    with log_container:
        recent_logs = engine.event_log[-40:]
        for entry in reversed(recent_logs):
            level = entry.get("level", "info")
            source = entry.get("source", "system")
            msg = entry.get("message", "")
            tick = entry.get("tick", 0)
            st.markdown(
                f'<div class="log-entry {level}"><b>[Tick {tick}]</b> <b>{source}</b>: {msg}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 第四行：能力晋升管道可视化 ──
    st.divider()
    st.subheader("🔗 能力晋升管道")

    pipeline_cols = st.columns(4)
    pipeline_stages = [
        ("💬 In-context\n(工作记忆)", "任务执行时的\n上下文材料"),
        ("📦 Out-of-context\n(缓冲区)", "待反思的\n原始记录"),
        ("📝 Experience\n(经验)", f"{sum(len(a.experiences) for a in engine.agents)} 条"),
        ("⭐ Skills\n(技能)", f"{sum(len(a.skills) for a in engine.agents)} 项"),
    ]

    for col, (title, detail) in zip(pipeline_cols, pipeline_stages):
        with col:
            st.markdown(f"**{title}**")
            st.caption(detail)

    # 选出有最多技能/经验的 Agent 做经验→技能晋升展示
    st.divider()
    st.caption("💡 晋升示例 (最近 5 条技能):")
    all_skills = []
    for a in engine.agents:
        for s in a.skills:
            all_skills.append((a.name, s))
    all_skills.sort(key=lambda x: x[1].promoted_at, reverse=True)
    for name, sk in all_skills[:5]:
        st.caption(f"  {name} | {sk.description[:50]}... (来源: {sk.promoted_from})")

    # ── 自动运行逻辑 ──
    if st.session_state.auto_running:
        time.sleep(st.session_state.tick_speed)
        engine.tick()
        st.rerun()


if __name__ == "__main__":
    main()
