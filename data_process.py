from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import json

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_custom_data():
    with open("data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def preprocess_function(examples, tokenizer, max_length=128):
    full_texts = [f"用户：{q}###助手：{a}" for q, a in zip(examples["question"], examples["answer"])]
    model_inputs = tokenizer(full_texts, truncation=True, max_length=max_length,
                             padding=False, return_attention_mask=False)

    labels = []
    for i, (q, a) in enumerate(zip(examples["question"], examples["answer"])):
        user_part = f"用户：{q}###助手："
        user_tokens = tokenizer(user_part, add_special_tokens=False)["input_ids"]
        input_ids = model_inputs["input_ids"][i]
        user_len = min(len(user_tokens), len(input_ids))
        label = [-100] * user_len + input_ids[user_len:]
        if len(label) != len(input_ids):
            label = [-100] * len(input_ids)
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset_dict = load_custom_data()
    tokenized_dataset = dataset_dict.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset_dict["train"].column_names
    )
    tokenized_dataset.save_to_disk("tokenized_data")
    print("数据处理完成")
