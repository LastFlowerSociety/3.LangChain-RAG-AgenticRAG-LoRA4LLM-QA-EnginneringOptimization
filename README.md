# 智能问答系统：LoRA微调 + RAG + Agentic RAG + 工程优化

这是一个完整的端到端智能问答系统，基于 **Qwen2.5-1.5B-Instruct** 模型，通过 **LoRA 微调** 让模型初步学习报文领域的知识，再结合 **RAG（检索增强生成）** 和 **Agentic RAG** 构建一个能够对话、检索、回答的智能助手。此外，还包含**缓存优化、异步调用、日志监控**等工程实践，让你体验从模型微调到应用落地的全流程。

> 💡 **说明**：LoRA 微调部分仅作为对 baseLM 与 LoRA 组合的尝试，数据量较小，效果有限，但可跑通流程。实际应用建议扩充数据或使用其他基座模型。

---

## 📁 项目结构
```
.
├── data.json # 微调数据（问答对）
├── data_process.py # 数据预处理，生成 tokenized_data
├── lora_finetune_cpu.py # LoRA 微调脚本（CPU 版，无需 GPU）
├── inference.py # 加载微调后的 LoRA 权重，合并模型并推理
├── langchain_rag.py # RAG 检索链（混合检索：BM25 + ChromaDB）
├── langchain_agent.py # Agentic RAG（多工具 + 记忆）
├── engineering_optimization.py # 工程优化：缓存、异步、日志
├── requirements.txt # 依赖包列表
└── README.md # 本文档
```
---

## 🚀 技术栈

- **模型**：Qwen2.5-1.5B-Instruct（HuggingFace）
- **微调**：LoRA (PEFT)，全精度 CPU 训练
- **数据**：自定义 JSON 格式问答对
- **RAG**：LangChain + ChromaDB + BM25（混合检索）
- **Agent**：LangChain LCEL 构建的工具调用链 + 对话记忆
- **工程优化**：LangChain 缓存、异步推理、日志监控
- **环境**：Python 3.10，Conda 管理环境

---

## 📦 安装与配置

### 1. 创建 conda 环境
```bash
conda create -n qwen_rag python=3.10 -y
conda activate qwen_rag
```
###2. 安装依赖
```bash
pip install -r requirements.txt
```
requirements.txt 内容如下：

```text
torch
transformers==4.38.2
accelerate==0.27.2
peft
datasets
langchain
langchain-community
langchain-chroma
chromadb
rank_bm25
sentence-transformers
```
💡 如果你有 GPU，可以安装 GPU 版 bitsandbytes，但 Windows 下建议用 CPU 版。

## 📊 数据准备
编辑 data.json，格式为：

```json
[
    {"question": "2025年3月北京地区报文编号是多少？", "answer": "2025BJ0301"},
    {"question": "报文2025BJ0301的核心内容是什么？", "answer": "北京地区2025年3月设备运行正常，无异常告警"},
    ...
]
```
提供了若干条示例数据（见项目内），你也可以根据自己的业务扩充。

🔧 使用流程
### 1. 数据预处理
```bash
python data_process.py
```
会生成 tokenized_data/ 目录，里面包含训练集和验证集。

### 2. LoRA 微调（CPU 版）
```bash
python lora_finetune_cpu.py
```
训练过程会输出 loss 和梯度范数，保存 LoRA 权重到 lora_model/。

⚠️ 说明：此部分仅作为对 baseLM 与 LoRA 组合的尝试，数据量较小，效果有限。若想获得更好效果，建议扩充数据或使用 GPU 版 bitsandbytes 进行 4-bit QLoRA 训练。

### 3. 合并模型并推理
```bash
python inference.py
```
会从 lora_model/ 加载 LoRA 权重，与基座模型合并，保存到 merged_model/。

测试两个推理配置：稳定模式（低 temperature）和随机模式（高 temperature）。
你可以修改 generate_answer 中的问题来测试自己的 query。

### 4. RAG 检索链（测试混合检索）
```bash
python langchain_rag.py
```

会演示：

文本分块

构建 BM25 + ChromaDB 混合检索器

使用 Qwen 模型（微调后）生成答案
如果还没有合并模型，可暂时用原始 Qwen，但推荐使用 merged_model。

### 5. Agentic RAG（多工具 + 记忆）
bash
python langchain_agent.py
这是一个更智能的对话系统：

工具1：rag_tool（查询报文信息）

工具2：format_check_tool（校验 JSON 格式）

自动管理对话记忆（最近 3 轮）

支持连续提问，例如“刚才的问题中，报文编号是什么？”

⚠️ 第一次运行 Agent 可能会慢，因为要加载模型。耐心等待就好。

### 6. 工程优化演示
```bash
python engineering_optimization.py
```
会展示：

缓存优化：第二次调用相同问题直接返回缓存，速度极快

异步调用：使用 ainvoke 提升并发能力

日志监控：记录请求到 rag_agent.log，便于排查

## 💬 结语
一起优化这个模型！

Happy Coding！
