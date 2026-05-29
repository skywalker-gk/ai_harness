import os
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# ================== 1. 定义全局状态 (State) ==================
class AnalysisState(TypedDict):
    """
    整个工作流中流转的数据结构
    """
    user_question: str       # 用户的原始问题
    generated_code: str      # AI 生成的 Python 代码
    execution_result: str    # 代码执行的结果或报错信息
    final_answer: str        # 最终给用户的回复
    max_retries: int         # 最大重试次数
    current_retry: int       # 当前重试次数

# ================== 2. 初始化大模型 ==================
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url=os.getenv("SILICONFLOW_BASE_URL"),
    temperature=0
)

# ================== 3. 定义工作流的节点 (Nodes) ==================

def code_generator_node(state: AnalysisState):
    """
    节点A：根据用户问题和之前的报错信息，生成或修正 Python 分析代码
    """
    state["current_retry"] += 1
    print(f"\n[节点A] 第 {state['current_retry']} 次尝试生成代码...")
    
    # 构造动态提示词
    error_context = ""
    if state.get("execution_result"):
        error_context = f"上一次执行的代码报错了，错误信息是：{state['execution_result']}。请修正代码。"
    
    system_prompt = f"""你是一个精通 Pandas 的数据分析师。请根据用户的问题编写一段纯 Python 代码。
假设内存中已经有一个名为 df 的 DataFrame。
要求：
1. 只输出代码块，不要包含任何解释性文字。
2. 将最终结果赋值给变量 result。
3. {error_context}
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"用户问题：{state['user_question']}")
    ]
    
    response = llm.invoke(messages)
    # 简单清洗，提取代码块
    code = response.content.replace("```python", "").replace("```", "").strip()
    state["generated_code"] = code
    return state

def code_executor_node(state: AnalysisState):
    """
    节点B：安全地执行生成的 Python 代码
    """
    print("[节点B] 正在沙箱环境中执行代码...")
    try:
        # ⚠️ 注意：在实际生产中，必须使用真正的 Docker 沙箱来执行 AI 代码。
        # 这里为了演示方便，使用 exec 并在受限的全局变量中运行。
        local_vars = {"df": None, "result": None} 
        # 模拟一个 df (实际项目中这里会传入真实数据)
        import pandas as pd
        local_vars['df'] = pd.DataFrame({"sales": [100, 200, 150], "region": ["A", "B", "C"]})
        
        exec(state["generated_code"], {}, local_vars)
        state["execution_result"] = f"执行成功，结果为：{local_vars.get('result')}"
    except Exception as e:
        state["execution_result"] = f"代码执行报错：{str(e)}"
    
    return state

def summarizer_node(state: AnalysisState):
    """
    节点C：将代码执行结果转化为人类可读的自然语言报告
    """
    print("[节点C] 正在生成最终分析报告...")
    system_prompt = "你是一个数据汇报专家。请根据代码的执行结果，用简练专业的语言回答用户的问题。"
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"问题：{state['user_question']}\n数据结果：{state['execution_result']}")
    ]
    response = llm.invoke(messages)
    state["final_answer"] = response.content
    return state

# ================== 4. 定义路由逻辑 (Conditional Edges) ==================
def should_continue(state: AnalysisState):
    """
    决定下一步去哪里：
    如果执行成功 -> 去总结节点 (summarizer)
    如果执行失败且没超过重试上限 -> 回去改代码 (code_generator)
    如果超过重试上限 -> 强制结束 (END)，避免死循环
    """
    if "报错" in state["execution_result"]:
        if state["current_retry"] < state["max_retries"]:
            return "regenerate" # 回去重新生成代码
        else:
            return "fail_end" # 放弃治疗
    else:
        return "success" # 执行成功，去总结

# ================== 5. 组装 LangGraph 工作流 ==================
def build_analysis_graph():
    workflow = StateGraph(AnalysisState)
    
    # 添加节点
    workflow.add_node("generate_code", code_generator_node)
    workflow.add_node("execute_code", code_executor_node)
    workflow.add_node("summarize", summarizer_node)
    
    # 设置入口点
    workflow.set_entry_point("generate_code")
    
    # 添加连线
    workflow.add_edge("generate_code", "execute_code") # 生成完代码，必然要去执行
    
    # 添加条件分支（根据执行结果决定走向）
    workflow.add_conditional_edges(
        "execute_code",
        should_continue,
        {
            "regenerate": "generate_code", # 报错且可重试 -> 回到生成节点
            "success": "summarize",        # 成功 -> 去总结
            "fail_end": END                # 彻底失败 -> 结束
        }
    )
    
    workflow.add_edge("summarize", END) # 总结完，流程结束
    
    # 编译成可执行的图
    app = workflow.compile()
    return app

# ================== 6. 测试运行 ==================
async def main():
    print("=== 启动 LangGraph 智能数据分析工作流 ===\n")
    app = build_analysis_graph()
    
    # 初始状态
    initial_state = {
        "user_question": "请计算 sales 列的平均值。",
        "generated_code": "",
        "execution_result": "",
        "final_answer": "",
        "max_retries": 3,
        "current_retry": 0
    }
    
    # invoke 会按照我们定义的图结构自动流转
    final_state = await app.ainvoke(initial_state)
    
    print("\n" + "="*30)
    print("🎉 最终分析报告：")
    print(final_state["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())
