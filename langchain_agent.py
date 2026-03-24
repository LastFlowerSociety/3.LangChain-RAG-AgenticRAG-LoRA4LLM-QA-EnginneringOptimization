# ========== 导入 ==========
import json
import re
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_classic.memory import ConversationBufferWindowMemory

# 从你的 RAG 模块导入组件
from langchain_rag import build_llm, build_retriever, split_text

# ========== 1. 初始化 RAG 组件 ==========
print("正在初始化 RAG 组件...")
chunks = split_text()
retriever = build_retriever(chunks)
llm = build_llm()
print("初始化完成。")

# ========== 2. 定义工具 ==========


@tool
def memory_tool(query: str) -> str:
    """从对话历史中查找与查询相关的信息。当用户问及之前对话内容时使用此工具。"""
    # 获取最近的历史消息
    history = memory.load_memory_variables({})["chat_history"]
    if not history:
        return "没有历史对话记录。"

    # 构造提示：让 LLM 根据历史回答查询
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    prompt_text = f"""根据以下对话历史，回答用户的问题。只输出答案，不要解释。

对话历史：
{history_text}

问题：{query}
答案："""

    # 调用 LLM（复用全局 llm，注意避免循环，这里只是文本生成）
    response = llm.invoke(prompt_text)
    return response.strip()


@tool
def rag_tool(question: str) -> str:
    """用于回答关于2025年3月报文的所有问题。输入是用户的问题，输出是答案+来源。"""
    if hasattr(retriever, "invoke"):
        docs = retriever.invoke(question)
    else:
        docs = retriever.get_relevant_documents(question)
    if not docs:
        return "未找到相关信息"
    source = "汇总文档"
    match = re.search(r"([A-Z]{2}\d{4})", question)
    if match:
        source = match.group(1)
    return f"{docs[0].page_content}（来源：{source}）"


@tool
def format_check_tool(text: str) -> str:
    """用于校验输出是否为JSON格式。输入是需要校验的文本，输出是校验结果。"""
    try:
        json.loads(text)
        return "格式正确"
    except:
        return "格式错误，请输出JSON格式"


tools = [rag_tool, format_check_tool]

# ========== 3. 初始化记忆 ==========
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,                     # 保留最近3轮对话
    return_messages=True
)

# ========== 4. 构建带记忆的链 ==========


def tools_description():
    lines = []
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
    return "\n".join(lines)


# 修改 prompt 模板，加入 {chat_history}
prompt = PromptTemplate.from_template(
    """你是一个智能助手，可以调用以下工具来回答问题。请输出一个 JSON 对象，包含工具名和输入参数。

可用工具：
{tools_description}

对话历史：
{chat_history}

用户当前的问题：{input}

请只输出一个 JSON 对象，格式：{{"tool": "工具名", "input": "输入内容"}}
不要输出任何其他内容，不要重复用户的问题或系统提示。"""
)


def parse_tool_call(output: str):
    # 1. 原有工具调用解析（保持不变）
    valid_tool_names = [t.name for t in tools]  # ["rag_tool", "format_check_tool"]
    json_pattern = r'\{.*?\}'
    matches = re.findall(json_pattern, output, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match)
            tool_name = data.get("tool")
            tool_input = data.get("input")
            if tool_name in valid_tool_names and tool_input is not None:
                return tool_name, tool_input
        except json.JSONDecodeError:
            continue

    # 2. 新增：如果没有有效工具调用，尝试从输出中提取最内层的 answer 字段
    decoder = json.JSONDecoder()
    idx = 0
    best_answer = None
    while idx < len(output):
        try:
            obj, end = decoder.raw_decode(output, idx)
            # 检查当前对象是否有 answer
            if isinstance(obj, dict) and "answer" in obj:
                best_answer = obj["answer"]   # 记录最新找到的 answer（后面的可能更内层）
            # 如果有 input 字段且是字符串，尝试解析内层 JSON
            if isinstance(obj, dict) and "input" in obj and isinstance(obj["input"], str):
                try:
                    inner = json.loads(obj["input"])
                    if isinstance(inner, dict) and "answer" in inner:
                        best_answer = inner["answer"]   # 内层 answer 优先级更高
                except:
                    pass
            idx = end
        except json.JSONDecodeError:
            idx += 1

    if best_answer is not None:
        return "直接回答", best_answer

    return None, None


def call_tool(tool_name: str, tool_input: str):
    if tool_name == "直接回答":
        return tool_input
    for t in tools:
        if t.name == tool_name:
            try:
                return t.func(tool_input)
            except Exception as e:
                return f"工具执行出错：{e}"
    return f"未找到工具：{tool_name}"


def debug_print(x):
    print("=== LLM原始输出 ===")
    print(repr(x))
    print("==================")
    return x


chain = (
    {"input": lambda x: x["input"], "tools_description": lambda _: tools_description(),
     "chat_history": lambda x: x["chat_history"]}
    | prompt
    | llm
    | StrOutputParser()
    | debug_print   # 加这行
    | parse_tool_call
    | (lambda x: call_tool(x[0], x[1]) if x[0] else f"无法解析工具调用，模型输出：{x[1] if x[1] else '无有效输出'}")
)
# 带记忆的调用函数


def run_with_memory(user_input: str):
    # 从 memory 加载历史（格式为 List[BaseMessage]）
    history = memory.load_memory_variables({})["chat_history"]
    # 将消息列表转为文本格式（简单拼接）
    if history:
        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history])
    else:
        history_text = "无"
    # 调用链
    result = chain.invoke({"input": user_input, "chat_history": history_text})
    # 更新记忆：保存用户输入和助手输出
    memory.save_context({"input": user_input}, {"output": result})
    return result


# ========== 5. 运行测试 ==========
if __name__ == "__main__":
    print("===== 测试工具调用链（带记忆） =====")

    # 第一轮
    user_input = "2025GZ0301的巡检结果是什么？请输出JSON格式，包含answer和source字段"
    print(f"用户：{user_input}")
    result = run_with_memory(user_input)
    print(f"助手：{result}\n")

    # 第二轮：依赖上下文
    user_input2 = "刚才的问题中，报文编号是什么？"
    print(f"用户：{user_input2}")
    result2 = run_with_memory(user_input2)
    print(f"助手：{result2}\n")

    # 第三轮：故障测试
    user_input3 = "随便说点什么，故意让工具调用失败"
    print(f"用户：{user_input3}")
    result3 = run_with_memory(user_input3)
    print(f"助手：{result3}")
