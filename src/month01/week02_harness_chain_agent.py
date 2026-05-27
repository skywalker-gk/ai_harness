import os
import asyncio
from dotenv import load_dotenv
from typing import List
import pandas as pd

# LangChain 核心组件
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.messages import HumanMessage

# 1. 加载环境变量
load_dotenv()

# ================== 知识点1 & 2: LCEL 链式编排与输出解析 ==================
async def demo_lcel_chain():
    print("--- 任务一：LCEL 链式编排与数据分析报告生成 ---")
    
    # 初始化大模型
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0.3
    )

    # 定义提示词模板 (Prompt)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的电商数据分析师。请根据用户提供的销售数据，生成一份简短的业务洞察报告。"),
        ("human", "销售数据如下：\n{sales_data}\n\n请给出你的分析：")
    ])

    # 定义输出解析器 (Output Parser)
    output_parser = StrOutputParser()

    # 使用 LCEL (管道符 |) 将组件串联成一条链
    analysis_chain = prompt | llm | output_parser

    # 模拟一段销售数据
    mock_sales_data = """
    日期, 产品, 销售额(万元)
    2026-05-01, 智能手机, 120
    2026-05-02, 智能手机, 135
    2026-05-03, 平板电脑, 80
    2026-05-04, 智能手机, 150
    2026-05-05, 平板电脑, 95
    """

    # 异步调用这条链 (ainvoke)
    print("正在生成数据分析报告...\n")
    report = await analysis_chain.ainvoke({"sales_data": mock_sales_data})
    print(f"AI 生成的分析报告：\n{report}\n")


# ================== 知识点3: Agents (智能体) 与 Tools (工具) ==================
# 准备一份 Pandas DataFrame 供 AI 操作
sales_df = pd.DataFrame({
    'date': ['2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05'],
    'product': ['手机', '手机', '平板', '手机', '平板'],
    'sales': [120, 135, 80, 150, 95]
})

# 使用 @tool 装饰器将普通 Python 函数包装成 AI 可调用的工具
@tool
def get_total_sales(product_name: str) -> float:
    """
    查询指定产品的总销售额。
    输入参数 product_name 必须是 '手机' 或 '平板'。
    """
    total = sales_df[sales_df['product'] == product_name]['sales'].sum()
    return total

@tool
def get_max_daily_sales(product_name: str) -> float:
    """
    查询指定产品的单日最高销售额。
    输入参数 product_name 必须是 '手机' 或 '平板'。
    """
    max_sales = sales_df[sales_df['product'] == product_name]['sales'].max()
    return max_sales

async def demo_data_agent():
    print("--- 任务二：构建 Pandas 数据分析智能体 ---")
    
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0
    )

    # 将我们定义的工具打包成一个列表
    tools = [get_total_sales, get_max_daily_sales]

    # 创建 Tool Calling Agent (基于函数调用的智能体)
    agent = create_tool_calling_agent(llm, tools, prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个拥有 Pandas 数据处理能力的 AI 助手。请准确使用工具回答用户的问题。"),
        ("human", "{input}")
    ]))

    # 创建 Agent 执行器 (AgentExecutor)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True 可以看到 AI 的思考过程

    # 向 Agent 提问，观察它如何自主调用工具
    questions = [
        "帮我查一下手机的总销售额是多少？",
        "平板电脑的单日最高销售额是多少？",
        "对比一下手机和平板的总销售额，哪个卖得更好？" # 测试 Agent 的多步推理能力
    ]

    for q in questions:
        print(f"\n用户提问：{q}")
        # Agent 的执行通常涉及多次 LLM 调用，这里使用同步 invoke 方便观察日志
        result = await agent_executor.ainvoke({"input": q})
        print(f"AI 最终回答：{result['output']}\n" + "-"*50)


# ================== 主程序入口 ==================
async def main():
    print("=== 开启 Harness Engineering 第二周实战：Chains 与 Agents ===\n")
    
    # 任务一：体验 LCEL 链式编排
    await demo_lcel_chain()
    
    # 任务二：体验 AI 智能体自主调用 Pandas 工具
    await demo_data_agent()

if __name__ == "__main__":
    asyncio.run(main())