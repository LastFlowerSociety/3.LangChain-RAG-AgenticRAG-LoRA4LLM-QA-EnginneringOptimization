from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基座模型+LoRA权重
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ========== 1：合并LoRA权重（部署用） ==========
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
# 加载LoRA权重
lora_model = PeftModel.from_pretrained(base_model, "./lora_model")
# 合并权重
merged_model = lora_model.merge_and_unload()
merged_model.save_pretrained("./merged_model")

# ========== 2：生成推理+问题排查 ==========


def generate_answer(question, use_best_config=True):
    prompt = f"用户：{question}###助手："
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # 生成参数配置（解决重复生成/格式错乱问题）
    generation_config = {
        "max_new_tokens": 64,
        "temperature": 0.1 if use_best_config else 1.0,  # 低temperature减少随机性
        "top_p": 0.9,
        "do_sample": False if use_best_config else True,  # 结构化生成关闭采样，核心
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    # 推理（关闭梯度，显存优化）
    with torch.no_grad():
        outputs = merged_model.generate(**inputs, **generation_config)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("###助手：")[-1]
    return answer


if __name__ == "__main__":
    # 测试正常生成
    print("正常生成（优化参数）：")
    print(generate_answer("2025年3月北京地区报文编号是多少？"))

    # 测试问题排查（故意用差的参数，模拟重复生成）
    print("\n问题生成（高temperature+采样）：")
    print(generate_answer("2025年3月北京地区报文编号是多少？", use_best_config=False))
