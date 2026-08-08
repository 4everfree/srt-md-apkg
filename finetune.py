import os
import pickle

if hasattr(pickle.Pickler, "_batch_setitems"):
    original_batch_setitems = pickle.Pickler._batch_setitems
    def patched_batch_setitems(self, items, obj=None):
        if obj is None:
            return original_batch_setitems(self, items)
        else:
            return original_batch_setitems(self, iter(items))
    pickle.Pickler._batch_setitems = patched_batch_setitems

import os
os.environ["HF_DATASETS_DISABLE_FINGERPRINTING"] = "1"
import datasets
import datasets.fingerprint
import datasets.arrow_dataset
datasets.fingerprint.generate_fingerprint = lambda *args, **kwargs: "dummy_fingerprint"
datasets.arrow_dataset.generate_fingerprint = lambda *args, **kwargs: "dummy_fingerprint"

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Конфигурация
max_seq_length = 4096 # Достаточно для субтитров и конспектов
dtype = None # Автоматический выбор (Float16 для старых GPU, BFloat16 для Ampere/Hopper)
load_in_4bit = True # Используем 4-bit квантование для экономии видеопамяти

# Загружаем базовую модель (можно поменять на Llama-3-8B, Mistral и т.д.)
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

print(f"Загрузка модели {model_id}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Настройка LoRA адаптеров (файнтюнинг только части весов)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Размерность LoRA (рекомендуется 16)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Оптимизация отключения dropout
    bias = "none",
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
)

# Загружаем наш подготовленный датасет
dataset_path = "train_dataset.jsonl"
print(f"Загрузка датасета {dataset_path}...")
import json
data_list = []
with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        data_list.append(json.loads(line))
        
# Форматируем и токенизируем тексты перед созданием датасета (чтобы не использовать SFTTrainer dataset_text_field)
tokenized_data = []
for row in data_list:
    text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    tokens = tokenizer(text, truncation=True, max_length=max_seq_length)
    tokenized_data.append({
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": tokens["input_ids"]
    })

from datasets import Dataset
dataset = Dataset.from_list(tokenized_data)

# Настройка параметров обучения
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Для тестов можно поставить 60. Для полного обучения лучше использовать num_train_epochs = 3
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Запуск обучения
print("Запуск файнтюнинга...")
trainer_stats = trainer.train()

# Сохранение обученной модели (сохраняются только веса LoRA)
save_path = "lora_model"
print(f"Сохранение адаптеров модели в {save_path}...")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Обучение завершено! Модель готова к использованию.")
