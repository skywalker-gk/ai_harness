import os
import asyncio
from dotenv import load_dotenv

# LangChain 核心组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # 轻量级本地向量数据库

# 1. 加载环境变量
load_dotenv()

# ================== 模拟企业私有数据 ==================
COMPANY_KNOWLEDGE = """
【华东区生鲜业务指标口径手册 V2.0】
1. 复购率定义：指在过去30天内，购买过2次及以上生鲜产品的独立用户数占总活跃用户数的比例。计算公式为：(30天内购买>=2次的UV / 30天总UV) * 100%。
2. 客单价(ATV)：统计周期内，生鲜品类的总GMV除以总订单数。注意：剔除退款订单和金额为0的测试订单。
3. 损耗率：生鲜商品在仓储、运输及销售过程中因腐烂、破损等原因造成的报废金额占当期进货总金额的比例。华东区的正常损耗率警戒线为 5%。
4. 履约成本：包含最后一公里配送费、分拣人工费及包装材料费。目前华东区单均履约成本控制在 8.5 元以内为优秀。
"""

# ================== 知识点1 & 2: 构建向量知识库 ==================
class CorporateKnowledgeBase:
    def __init__(self):
        # 初始化硅基流动的 Embedding 模型 (用于将文字转为向量)
        self.embeddings = OpenAIEmbeddings(
            model="BAAI/bge-m3", # 推荐使用 bge-m3，中文语义理解极佳且免费额度友好
            openai_api_key=os.getenv("SILICONFLOW_API_KEY"),
            openai_api_base=os.getenv("SILICONFLOW_BASE_URL")
        )
        
        # 初始化本地向量数据库 (持久化保存在 ./chroma_db 文件夹)
        self.vector_store = Chroma(persist_directory="./chroma_db", embedding_function=self.embeddings)
    
    def ingest_data(self, raw_text: str):
        """将长文本切片并注入向量数据库"""
        print("正在对业务文档进行切片和向量化入库...")
        
        # 文本切分器：按字符切分，每块500字，重叠50字以保持上下文连贯
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(raw_text)
        
        # 将切片存入向量库
        self.vector_store.add_texts(chunks)
        print(f"成功入库 {len(chunks)} 个知识片段。\n")

    async def retrieve_context(self, query: str, k=2):
        """根据用户的问题，去向量库里检索最相关的知识片段"""
        # similarity_search 会返回与 query 向量距离最近的 k 个文档对象
        docs = self.vector_store.similarity_search(query, k=k)
        # 提取出纯文本内容并用换行符拼接
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

# ================== 知识点3: RAG 问答链 ==================
async def rag_qa_chain(kb: CorporateKnowledgeBase, user_question: str):
    """执行完整的 RAG 流程：检索 -> 组装 Prompt -> 生成答案"""
    
    # 1. 检索阶段：先去知识库里捞取相关背景信息
    retrieved_context = await kb.retrieve_context(user_question)
    
    # 2. 生成阶段：设计带有“防幻觉”约束的 Prompt
    system_prompt = f"""
    你是一个严谨的企业内部数据分析师助手。
    请严格根据以下【参考背景资料】来回答用户的问题。
    如果【参考背景资料】中没有包含回答问题所需的信息，请直接回答：“抱歉，当前的企业知识库中未找到相关信息。”
    严禁利用你自身的通用训练数据进行编造或推测。

    【参考背景资料】：
    {retrieved_context}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    # 初始化大模型
    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-V3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0 # 必须设为0，保证回答的绝对客观性
    )

    chain = prompt | llm
    
    print(f"用户提问: {user_question}")
    print(f"AI 检索到的背景资料:\n{retrieved_context}\n{'-'*40}")
    
    # 3. 获取最终回复
    response = await chain.ainvoke({"question": user_question})
    print(f"AI 最终回复: {response.content}\n{'='*50}\n")

# ================== 主程序入口 ==================
async def main():
    print("=== 开启 Harness Engineering 第三周实战：RAG 企业知识库 ===\n")
    
    # 初始化知识库并注入数据
    kb = CorporateKnowledgeBase()
    kb.ingest_data(COMPANY_KNOWLEDGE)

    # --- 测试一：询问有明确答案的业务指标 ---
    await rag_qa_chain(kb, "请问华东区生鲜业务的复购率具体是怎么定义的？计算公式是什么？")

    # --- 测试二：询问另一个指标，考验语义匹配能力 ---
    await rag_qa_chain(kb, "我们华东区的单均履约成本控制目标是多少？")

    # --- 测试三：询问知识库中不存在的内容，测试防幻觉机制 ---
    await rag_qa_chain(kb, "华南区的蔬菜损耗率警戒线是多少？")

if __name__ == "__main__":
    asyncio.run(main())