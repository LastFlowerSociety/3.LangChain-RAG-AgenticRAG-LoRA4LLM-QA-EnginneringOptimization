from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import HuggingFacePipeline
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List
from collections import defaultdict
import math


from typing import Any, List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from collections import defaultdict


from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
from collections import defaultdict

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from typing import List


class SimpleBM25Retriever:
    """简易 BM25 检索器，直接使用 rank_bm25 实现"""

    def __init__(self, chunks: List[str]):
        # 分词：这里简单按空格分割，中文可改进（但你的文档示例以中文为主，空格分隔效果有限）
        # 为了简单，先用空格分，实际可换成 jieba，但先保证能跑通
        tokenized_chunks = [chunk.split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.chunks = chunks

    def get_relevant_documents(self, query: str, k: int = 3) -> List[Document]:
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        # 获取 top k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [Document(page_content=self.chunks[i]) for i in top_indices]


class HybridRetriever(BaseRetriever):
    bm25_retriever: object = Field(description="BM25 retriever instance (with get_relevant_documents)")
    vectorstore: object = Field(description="Chroma vectorstore instance (with similarity_search)")
    weights: List[float] = Field(default=[0.4, 0.6])
    k: int = Field(default=3)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # BM25 检索
        bm25_docs = self.bm25_retriever.get_relevant_documents(query, k=self.k*2)
        # 向量检索
        vector_docs = self.vectorstore.similarity_search(query, k=self.k*2)

        # 加权排名融合
        scores = defaultdict(float)
        for rank, doc in enumerate(bm25_docs):
            scores[doc.page_content] += self.weights[0] * (1 / (rank + 1))
        for rank, doc in enumerate(vector_docs):
            scores[doc.page_content] += self.weights[1] * (1 / (rank + 1))

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:self.k]
        doc_map = {doc.page_content: doc for doc in bm25_docs + vector_docs}
        result = [doc_map[content] for content, _ in sorted_docs if content in doc_map]
        return result
# ========== 1：文本分块 ==========


def split_text():
    document = """
    2025年3月全国报文汇总：
    北京地区：编号2025BJ0301，2025年3月10日巡检，设备运行正常，无异常告警。
    上海地区：编号2025SH0301，2025年3月10日巡检，设备巡检正常，无异常。
    广州地区：编号2025GZ0301，2025年3月12日巡检，发现1处小故障，已修复，无后续异常。
    报文编号格式：四位年份+两位地区缩写+四位月份日期。
    异常报文处理流程：发现异常→记录编号→定位问题→修复→更新报文状态。
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=16,
        separators=["\n", "。", "，"]
    )
    chunks = text_splitter.split_text(document)
    return chunks

# ========== 2：向量化 + 混合检索（BM25 + ChromaDB） ==========


def build_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    # 向量库（Chroma）
    chroma_db = Chroma.from_texts(chunks, embeddings, collection_name="rag_demo")

    # 自定义 BM25 检索器
    bm25_retriever = SimpleBM25Retriever(chunks)

    # 混合检索器（稍后修改）
    hybrid = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vectorstore=chroma_db,
        weights=[0.4, 0.6],
        k=3
    )
    return hybrid
# ========== 3：集成模型到 LangChain ==========


def build_llm():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 这里用你自己的模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


# ========== 4：RAG 链路整合 ==========
if __name__ == "__main__":
    chunks = split_text()
    retriever = build_retriever(chunks)
    llm = build_llm()

    # 构建 Prompt 模板
    prompt_template = """
    基于以下上下文回答问题，严格按照格式输出，只回答问题本身，不要额外内容：
    上下文：{context}
    问题：{input}
    答案：
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "input"]
    )

    # 创建文档组合链
    combine_docs_chain = create_stuff_documents_chain(prompt=prompt, llm=llm)

    # 创建检索链
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 测试 RAG
    result = rag_chain.invoke({"input": "2025GZ0301 的巡检结果是什么？"})
    print("问题：", "2025GZ0301 的巡检结果是什么？")
    print("答案：", result["answer"])
    print("检索的源文档：", [doc.page_content for doc in result["context"]])
