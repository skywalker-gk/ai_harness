import os
import asyncio
from dotenv import load_dotenv
from typing import List, Dict

# LangChain 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver # 用于持久化状态/记忆

# 1. 加载环境变量
load_dotenv()

# ================== 知识点3: 定义 AI 可调用的真实工具 ==================
@tool
def calculate_growth_rate(current_value: float, previous_value: float) -> str:
    """
    计算环比增长率。当用户询问增长率、涨跌幅时使用此工具。
    参数:
        current_value: 当前周期的数值
        previous_value: 上一个周期的数值
    返回:
        格式化的增长百分比字符串
    """
    if previous_value == 0:
        return "无法计算（上期基数为0）"
    rate = ((current_value - previous_value) / previous_value) * 100
    return f"{rate:.2f}%"

@tool
def query_mock_database(region: str) -> str:
    """
    模拟查询公司内部数据库。当用户询问具体区域的业务数据时使用此工具。
    参数:
        region: 区域名称，如 '华东'、'华北'、'华南'
    返回:
        该区域的最新销售额数据
    """
    mock_db = {
        "华东": 158.9,
        "华北": 98.5,
        "华南": 120.3
    }
    value = mock_db.get(region, 0)
    return f"{region}区的最新销售额为 {value} 万元"

# ================== 知识点1 & 2: 搭建带记忆的 Agent ==================
async def build_analyst_agent():
    """构建并返回一个带有记忆功能的智能数据分析 Agent"""
    
    # 初始化大模型
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0
    )

    # 准备工具列表
    tools = [calculate_growth_rate, query_mock_database]

    # 设计 System Prompt (Harness 的缰绳)
    system_prompt = """
    你是一个专业的企业级数据分析师。
    你的职责是利用手中的工具（query_mock_database, calculate_growth_rate）来精准回答用户的问题。
    如果工具返回了数据，请结合数据给出简练的业务洞察。
    注意：在回答时，请务必记住用户之前的对话上下文。
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"), # 占位符，用于插入历史记忆
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # 占位符，用于存放 Agent 思考过程
    ])

    # 创建 Tool Calling Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 使用 MemorySaver 作为检查点保存器，实现多轮对话的状态记忆
    memory = MemorySaver()
    
    # 封装为 AgentExecutor，并绑定内存
    agent_executor = AgentExecutor(agent=agent, tools=tools, checkpointer=memory)
    
    return agent_executor, memory

# ================== 主程序入口：模拟多轮对话 ==================
async def main():
    print("=== 开启 Harness Engineering 第二周实战：带记忆的工具调用 Agent ===\n")
    
    agent_executor, memory = await build_analyst_agent()
    
    # 配置线程ID，相当于给这次对话分配一个独立的“会话房间”
    config = {"configurable": {"thread_id": "analyst_session_001"}}

    # --- 第一轮对话：查询数据 ---
    print("--- 第一轮对话 ---")
    input_1 = {"input": "帮我查一下华东区的销售额是多少？"}
    result_1 = await agent_executor.ainvoke(input_1, config=config)
    print(f"AI回复: {result_1['output']}\n")

    # --- 第二轮对话：利用记忆 + 调用计算工具 ---
    print("--- 第二轮对话 ---")
    # 注意：这里我们没有重复提“华东区”，也没有提供具体的数字，考验 AI 的记忆和工具组合能力
    input_2 = {"input": "那华北区呢？另外，对比刚才查到的华东区数据，华北区的增长率（假设华北是上期，华东是本期）大概是多少？"}
    result_2 = await agent_executor.ainvoke(input_2, config=config)
    print(f"AI回复: {result_2['output']}\n")

if __name__ == "__main__":
    asyncio.run(main())