from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk
import torch

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# 加载 tokenizer 和模型（全精度 CPU 模式）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cpu",
    trust_remote_code=True,
    torch_dtype=torch.float32
)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 加载预处理数据
tokenized_dataset = load_from_disk("tokenized_data")

# DataCollator 自动处理 padding 和 labels 对齐
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100  # 忽略 labels 中的 pad token
)

training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,           # 调低学习率
    num_train_epochs=100,
    warmup_steps=10,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=5,
    save_strategy="steps",
    save_steps=5,
    load_best_model_at_end=True,
    report_to="none",
    max_grad_norm=0.1,            # 梯度裁剪
    fp16=False                    # CPU 模式
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    data_collator=data_collator,
)

if __name__ == "__main__":
    # 打印一条样本检查
    sample = tokenized_dataset["train"][0]
    print("=" * 40)
    print("原始输入：", tokenizer.decode(sample["input_ids"]))
    print("模型需要生成的答案：", tokenizer.decode([tid for tid in sample["labels"] if tid != -100]))
    print("=" * 40)
    trainer.train()
    model.save_pretrained("./lora_model")
    print("LoRA 微调完成")
