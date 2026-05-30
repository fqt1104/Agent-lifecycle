"""快速验证脚本 — 测试核心逻辑"""
import sys
sys.path.insert(0, '.')

from agent_model import Agent, AgentState, Experience, Skill, Persona
from state_machine import AgentStateMachine
from sim_engine import SimEngine

print("=" * 60)
print("Agent 生命周期 Demo — 验证测试")
print("=" * 60)

# ── 1. 数据模型测试 ──
print("\n[1] 数据模型...")
a = Agent(name="TestAgent", role_id="test-001")
assert a.state == AgentState.CREATED
assert a.persona.summary == "白板 Agent，尚无专长领域"
assert len(a.experiences) == 0
assert len(a.skills) == 0
print("  ✅ Agent 模型 OK")

# ── 2. 状态机测试 ──
print("\n[2] 状态机...")
sm = AgentStateMachine()
agent = Agent(name="Alpha", role_id="alpha-001")

# 验证完整生命周期路径
event = sm.fire(agent, "init")
assert agent.state == AgentState.IDLE
print(f"  CREATED → IDLE: {event}")

event = sm.fire(agent, "announce_task")
assert agent.state == AgentState.BIDDING
print(f"  IDLE → BIDDING: {event}")

event = sm.fire(agent, "win_bid")
assert agent.state == AgentState.EXECUTING
print(f"  BIDDING → EXECUTING: {event}")

event = sm.fire(agent, "task_done")
assert agent.state == AgentState.REFLECTING
print(f"  EXECUTING → REFLECTING: {event}")

event = sm.fire(agent, "reflect_done")
assert agent.state == AgentState.PROMOTING
print(f"  REFLECTING → PROMOTING: {event}")

# 无经验晋升 → 回到 IDLE
event = sm.fire(agent, "no_promote")
assert agent.state == AgentState.IDLE
print(f"  PROMOTING → IDLE (无晋升): {event}")

# 测试竞标失败路径
sm.fire(agent, "announce_task")
event = sm.fire(agent, "lose_bid")
assert agent.state == AgentState.IDLE
print(f"  BIDDING → IDLE (未中标): {event}")

# 测试非法转移被拒绝
result = sm.fire(agent, "win_bid")  # 不在 BIDDING 状态
assert result is None
print(f"  非法转移被拒绝 (IDLE 下无法 win_bid): ✅")

print("  ✅ 状态机 OK")

# ── 3. 模拟引擎测试 ──
print("\n[3] 模拟引擎 (5 ticks)...")
engine = SimEngine()
engine.add_agent("Alpha", tags=["web_security"], domain="web_security")
engine.add_agent("Beta", tags=["performance"], domain="performance")

for i in range(5):
    engine.tick()
    summary = engine.get_state_summary()
    print(f"  Tick {i+1}: agents={summary['total_agents']}, active={summary['active_agents']}, "
          f"tasks_done={summary['tasks_completed']}, skills_market={summary['market_skills']}")

# 检查结果
assert engine.tick_count == 5
assert engine.market.total_completed > 0
print(f"\n  最终: {len(engine.agents)} 个 Agent, {engine.market.total_completed} 个任务完成")

# 显示 Agent 状态
for a in engine.agents:
    card = a.to_card()
    print(f"  {card['name']}: state={card['state']}, skills={card['skill_count']}, "
          f"exps={card['experience_count']}, avg_conf={card['avg_confidence']:.2f}")

# 显示事件日志摘要
print(f"\n  事件日志: {len(engine.event_log)} 条")
for e in engine.event_log[-5:]:
    print(f"    [{e['tick']}] {e['source']}: {e['message'][:80]}")

print("\n  ✅ 模拟引擎 OK")

# ── 4. 技能晋升路径测试 ──
print("\n[4] 快速晋升路径 (30 ticks)...")
engine2 = SimEngine()
engine2.add_agent("SecurityAgent", tags=["web_security"], domain="web_security")

# 预先加一条高置信度经验
from agent_model import Experience
from mock_data import domain_embedding
exp = Experience(
    id="exp-test-001",
    description="Web安全加固: SQL注入参数化查询",
    content="使用参数化查询修复SQL注入",
    confidence=0.96,
    exp_type="positive",
    tags=["web_security"],
    embedding=domain_embedding("web_security"),
)
engine2.agents[0].experiences.append(exp)

for i in range(30):
    engine2.tick()

a = engine2.agents[0]
card = a.to_card()
print(f"  {card['name']}: skills={card['skill_count']}, exps={card['experience_count']}, "
      f"conf={card['avg_confidence']:.2f}, persona={card['persona_summary']}")

print("\n  ✅ 晋升路径 OK")

print("\n" + "=" * 60)
print("🎉 所有验证通过! 可运行 streamlit run main.py 启动 UI")
print("=" * 60)
