from langchain_classic.cache import InMemoryCache
from langchain_core.globals import set_llm_cache
import logging
import asyncio
from langchain_rag import build_llm

# ========== 考点1：缓存优化（减少重复LLM调用） ==========


def cache_optimization():
    # 设置内存缓存
    set_llm_cache(InMemoryCache())
    llm = build_llm()

    # 第一次调用（无缓存）
    print("第一次调用：")
    print(llm.invoke("2025BJ0301的巡检时间是什么？"))   # 改用 .invoke()

    # 第二次调用（有缓存）
    print("\n第二次调用（缓存）：")
    print(llm.invoke("2025BJ0301的巡检时间是什么？"))   # 同上

# ========== 考点2：异步调用（提升并发） ==========


async def async_inference(question):
    llm = build_llm()
    result = await llm.ainvoke(question)   # 改用异步 invoke
    return result

# ========== 考点3：日志监控（链路可观测） ==========


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("rag_agent.log"), logging.StreamHandler()]
    )
    logger = logging.getLogger("rag_agent")
    return logger


if __name__ == "__main__":
    # 缓存优化
    cache_optimization()

    # 异步调用
    async def main():
        result = await async_inference("2025SH0301的巡检结果是什么？")
        print("\n异步调用结果：", result)
    asyncio.run(main())

    # 日志监控
    logger = setup_logger()
    logger.info("RAG Agent系统启动")
    logger.info("用户提问：2025GZ0301的巡检结果是什么？")
    logger.error("模拟格式校验失败：输出非JSON格式")
