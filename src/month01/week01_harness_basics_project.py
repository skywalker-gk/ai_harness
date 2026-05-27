from enum import Enum
import os
import asyncio
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# 1. 加载环境变量
load_dotenv()

# ================== 知识点1 & 2: 异步调用与 ChatModel 封装 ==================
class SiliconFlowClient:
    """
    封装基于硅基流动的异步大模型客户端
    """
    def __init__(self, model_name="deepseek-ai/DeepSeek-V3"):
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=os.getenv("SILICONFLOW_API_KEY"),
            base_url=os.getenv("SILICONFLOW_BASE_URL"),
            temperature=0 # 数据分析要求严谨，将温度设为0以减少随机性
        )

    async def ask(self, prompt: str):
        """异步发送单条指令"""
        print(f"正在向 AI 提问: {prompt[:30]}...")
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    async def batch_ask(self, prompts: List[str]):
        """异步并发发送多条指令 (AsyncIO 的威力)")"""
        tasks = [self.ask(prompt) for prompt in prompts]
        # gather 会将多个协程打包并发执行
        results = await asyncio.gather(*tasks)
        return results

# ================== 知识点3: 提示词模板 ==================
def get_data_summary_prompt(data_desc: str):
    """动态生成针对某段数据的分析指令"""
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一个资深的数据分析师。请用极其简练的语言总结用户提供的数据描述。"),
        ("human", "请从以下文本中抽取指标名称、数值和趋势，并给出中文建议：{data_desc}")
    ])
    # format_messages 返回的是可以直接传给 ChatModel 的消息列表
    return template.format_messages(data_desc=data_desc)

# ================== 知识点4: 结构化输出 (Pydantic) ==================
# class TrendType(str, Enum):
#     UP = "上升"
#     DOWN = "下降"
#     FLAT = "持平"

class DataInsight(BaseModel):
    """定义我们希望 AI 输出的标准 JSON 结构"""
    model_config = ConfigDict(strict=True)

    metric_name: str = Field(description="指标名称，必须提取原文中的中文名称，例如：销售额、日活")
    value: float = Field(description="指标的数值")
    trend: Literal["上升", "下降", "持平"] = Field(description="趋势判断")
    suggestion: str = Field(description="基于该指标的一句话业务建议，不要复述原文数据")
    confidence_score: float = Field(description="AI 对结果的可信度评分")

async def get_structured_analysis(client: SiliconFlowClient):
    """演示如何让 AI 输出符合 Pydantic 规范的 JSON"""
    # with_structured_output 是 LangChain 提供的强大功能，自动处理 Prompt 工程和 JSON 解析
    structured_llm = client.llm.with_structured_output(DataInsight, include_raw=True)
    
    raw_text = "上周华东区生鲜品类销售额达到 158.9 万元，环比前一周增长了 12.5%，表现非常亮眼。"
    raw_messages = get_data_summary_prompt(raw_text)
    
    print("\n正在让 AI 进行结构化信息抽取...")
    # invoke 会自动把结果转换成 DataInsight 对象
    result_dict = await structured_llm.ainvoke(raw_messages)
    insight: DataInsight = result_dict['parsed']
    return insight

# ================== 主程序入口 ==================
async def main():
    print("=== 开启 Harness Engineering 第一周实战 ===\n")
    
    # 初始化客户端
    ai_client = SiliconFlowClient()

    # # --- 任务一：体验异步并发 ---
    # print("--- 任务一：异步并发提问测试 ---")
    # start_time = time.time()
    # questions = [
    #     "一句话解释什么是 Transformer？",
    #     "Pandas 中 merge 和 join 的区别是什么？",
    #     "列出三个常用的时间序列预测模型。"
    # ]
    # answers = await ai_client.batch_ask(questions)
    # for i, ans in enumerate(answers):
    #     print(f"问题 {i+1} 的回答: {ans}\n")
    # print(f"异步并发总耗时: {time.time() - start_time:.2f} 秒\n")

    # --- 任务二：结构化信息抽取 (Harness 的核心) ---
    print("--- 任务二：结构化数据抽取测试 ---")
    result = await get_structured_analysis(ai_client)
    
    print("AI 提取的结构化结果为：")
    print(f"指标名称: {result.metric_name}")
    print(f"数值: {result.value}")
    print(f"趋势: {result.trend}")
    print(f"建议: {result.suggestion}")
    print(f"置信度: {result.confidence_score}\n")
    
    # 证明它真的是一个 Python 对象，可以直接参与后续计算
    if result.trend == "上升":
        print("\n[系统自动触发] 检测到趋势上升，已自动将该指标加入本周重点监控报表！")

if __name__ == "__main__":
    # 在 Windows 上运行 asyncio 的标准入口
    asyncio.run(main())