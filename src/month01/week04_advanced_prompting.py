import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载环境变量
load_dotenv()

# ================== 知识点3: 定义严格的 Pydantic 数据结构 ==================
class BusinessMetric(BaseModel):
    """定义我们希望 AI 从文本中提取的标准业务指标结构"""
    metric_name: str = Field(description="指标名称，例如：销售额、日活(DAU)、复购率")
    value: float = Field(description="指标的具体数值，如果是百分比请转换为小数，如5%转为0.05")
    unit: str = Field(description="数值的单位，例如：万元、人、%")
    time_period: str = Field(description="该指标所属的时间周期，例如：上周、Q1、2024年5月")
    trend_comment: Optional[str] = Field(default=None, description="如果原文提到了趋势变化(如环比增长)，在此记录")

class WeeklyReportExtraction(BaseModel):
    """包裹类，用于一次性提取报告中的多个指标"""
    report_summary: str = Field(description="对这份周报核心内容的简短一句话总结")
    metrics: List[BusinessMetric] = Field(description="从报告中提取出的所有关键业务指标列表")

# ================== 知识点1 & 2: 设计带思维链(CoT)的结构化 Prompt ==================
async def extract_metrics_from_text(raw_report: str):
    """演示如何利用高级提示词工程，让 AI 输出完美的 JSON"""
    
    # 初始化大模型
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0 # 数据提取要求绝对严谨，温度设为0
    )

    # 使用 with_structured_output 绑定 Pydantic 模型
    # LangChain 会自动在后台为你生成极其复杂的 Function Calling Prompt
    structured_llm = llm.with_structured_output(WeeklyReportExtraction)
    
    # 设计带有 CoT 引导的系统指令
    system_instruction = """
    你是一位拥有10年经验的首席数据分析师。你的任务是从杂乱的周报文本中提取核心业务指标。
    
    【分析步骤】：
    1. 首先，通读全文，识别出文中提到的所有定量数据（金额、人数、百分比等）。
    2. 其次，判断这些数据是否属于核心业务指标（如销售额、用户量、转化率等），过滤掉无关数据。
    3. 接着，仔细核对每个指标对应的时间周期和单位。如果原文是"增长了5%"，请将其转化为趋势备注，并确保主数值提取准确。
    4. 最后，严格按照定义的 JSON Schema 格式输出结果，不要包含任何 Markdown 标记或额外的解释文字。
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "请分析以下周报内容并提取指标：\n{raw_report}")
    ])

    chain = prompt | structured_llm
    
    print("正在启动高级提示词引擎进行数据清洗...")
    result: WeeklyReportExtraction = await chain.ainvoke({"raw_report": raw_report})
    return result

# ================== 主程序入口 ==================
async def main():
    print("=== 开启 Harness Engineering 第四周实战：高级提示词与结构化输出 ===\n")
    
    # 模拟一段非常口语化、杂乱的真实业务周报
    messy_business_report = """
    老大好，汇报一下上周华东区生鲜的情况。
    整体来看势头不错，上周我们的总销售额大概冲到了 158.9 万元左右，相比前一周大概涨了有 12.5% 吧。
    用户方面，日活跃用户数（DAU）稳定在 5.2 万人。不过有个问题，因为最近梅雨季节，物流有点慢，导致我们的订单履约成本有点高，平均每单达到了 9.2 块钱，超过了咱们 8.5 块的优秀线。
    另外，新上的樱桃品类卖得挺好，复购率在上个月达到了 35%。
    以上，请审阅。
    """

    # 执行提取
    extraction_result = await extract_metrics_from_text(messy_business_report)
    
    # 打印 Python 对象结果
    print("\n✅ AI 成功提取的结构化数据对象：")
    print(f"报告摘要: {extraction_result.report_summary}")
    print("-" * 50)
    
    for i, metric in enumerate(extraction_result.metrics, 1):
        print(f"指标 {i}:")
        print(f"  名称: {metric.metric_name}")
        print(f"  数值: {metric.value} ({metric.unit})")
        print(f"  周期: {metric.time_period}")
        print(f"  备注: {metric.trend_comment}")
        print("-" * 50)

    # 证明它可以无缝对接 Pandas
    print("\n🚀 假设将此结果传入 Pandas：")
    import pandas as pd
    df = pd.DataFrame([m.dict() for m in extraction_result.metrics])
    print(df[['metric_name', 'value', 'unit']])

if __name__ == "__main__":
    asyncio.run(main())