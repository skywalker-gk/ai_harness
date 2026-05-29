import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser

# 1. 加载环境变量
load_dotenv()

# ================== 初始化大模型 ==================
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0.2 # 稍微增加一点温度，让生成的报告文字更自然
)

# ================== 知识点1 & 3: 定义数据结构与解析器 ==================
class MetricData(BaseModel):
    """第一步：从自然语言中提取出的核心指标"""
    region: str = Field(description="地区名称")
    metric_name: str = Field(description="指标名称")
    value: float = Field(description="具体数值")
    period: str = Field(description="统计周期，如：上周、本月")

parser = PydanticOutputParser(pydantic_object=MetricData)

# ================== 知识点1: 构建 SequentialChain (顺序链) ==================
async def build_analysis_chain():
    """构建一个两步走的自动化分析流水线"""
    
    # --- 第一步：信息抽取链 ---
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个精准的数据提取助手。请从用户的描述中提取关键业务指标，并以JSON格式输出。\n{format_instructions}"),
        ("human", "{user_input}")
    ])
    
    # 注入 Pydantic 的格式指令到 System Prompt 中
    format_instructions = parser.get_format_instructions()
    extraction_chain = LLMChain(
        llm=llm, 
        prompt=extraction_prompt.partial(format_instructions=format_instructions),
        output_parser=parser,
        output_key="extracted_data"
    )

    # --- 第二步：报告生成链 ---
    report_prompt = ChatPromptTemplate.from_template("""
    你是一名资深商业分析师。基于以下提取到的数据，写一段简短专业的业务洞察报告。
    数据内容：{extracted_data}
    """)
    report_chain = LLMChain(
        llm=llm,
        prompt=report_prompt,
        output_key="final_report"
    )

    # --- 组装顺序链 ---
    # input_variables: 整个链条对外暴露的输入接口
    # output_variables: 整个链条最终输出的结果
    overall_chain = SequentialChain(
        chains=[extraction_chain, report_chain],
        input_variables=["user_input"],
        output_variables=["extracted_data", "final_report"],
        verbose=True # 开启verbose，可以在控制台看到每一步的执行过程
    )
    return overall_chain

# ================== 知识点2: 引入记忆系统 (Memory) ==================
async def demo_with_memory():
    """演示如何让 AI 记住上一轮的对话"""
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    # 创建一个带记忆的简单链
    prompt_with_memory = ChatPromptTemplate.from_messages([
        ("system", "你是一个数据分析助手。结合聊天记录回答问题。"),
        ("human", "{input}")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt_with_memory, memory=memory)
    
    print("\n=== 开启带记忆的对话测试 ===")
    # 第一次提问
    resp1 = await chain.ainvoke({"input": "帮我看下华东区上个月的销售额是500万"})
    print(f"AI回复1: {resp1['text']}\n")
    
    # 第二次提问（不提华东区，看它是否能记住上下文）
    resp2 = await chain.ainvoke({"input": "那华南区呢？假设是300万"})
    print(f"AI回复2: {resp2['text']}\n")

# ================== 主程序入口 ==================
async def main():
    print("=== 开启 Harness Engineering 第二周实战：Chains & Memory ===\n")
    
    # --- 任务一：运行顺序链 ---
    print("--- 任务一：自动化数据分析流水线 ---")
    analysis_chain = await build_analysis_chain()
    
    user_query = "上个季度北京地区的用户活跃度(DAU)达到了日均12.5万人，表现优异。"
    result = await analysis_chain.ainvoke({"user_input": user_query})
    
    print("\n【最终生成的业务洞察报告】：")
    print(result["final_report"])
    
    # --- 任务二：运行记忆测试 ---
    await demo_with_memory()

if __name__ == "__main__":
    asyncio.run(main())
