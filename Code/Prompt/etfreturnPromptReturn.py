import os
import torch
import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotenv import load_dotenv

load_dotenv()

# Load prompt-response pairs from your logic
def get_training_pairs():
    # Placeholder example — replace with your ETF prompt/response generator
    return [
        {"prompt": "What is the return of HDFC Gold ETF in 2024?", "response": "The return of HDFC Gold ETF in 2024 was 12.43%."},
        {"prompt": "What was the average return of gold ETFs in 2023?", "response": "The average return of gold ETFs in 2023 was 15.62%."},
    ]

def format_lm(example, tokenizer):
    return tokenizer(
        f"<s>[Prompt] {example['prompt']} [/Prompt] [Answer] {example['response']} [/Answer]</s>",
        truncation=True,
        padding="max_length",
        max_length=256
    )

# Load base model in 4-bit
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
token = os.getenv("HUGGINGFACE_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
tokenizer.pad_token = tokenizer.eos_token

print("[INFO] Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    token=token
)
model = prepare_model_for_kbit_training(model)

# Apply LoRA
print("[INFO] Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Prepare dataset
print("[INFO] Formatting dataset...")
pairs = get_training_pairs()
dataset = Dataset.from_list(pairs)
# Step 1: Add "text" field formatted for Causal LM
def format_lm(example):
    return {
        "text": f"<s>[Prompt] {example['prompt']} [/Prompt] [Answer] {example['response']} [/Answer]</s>"
    }

dataset = dataset.map(format_lm)

# Step 2: Tokenize the "text" field
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=["prompt", "response", "text"]  )

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training config
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
output_dir = f"Models/Training/Mistral-LoRA-{timestamp}"
logging_dir = f"logs/etfreturnPromptReturn-{timestamp}"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    num_train_epochs=2,
    save_steps=200,                  
    save_total_limit=2,
    max_steps=200,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=20,
    logging_dir=logging_dir,
    fp16=True,
    report_to="none"
)

# Trainer setup
print("[INFO] Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # For testing: same data
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Save final LoRA model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"[INFO] Training complete. Model saved to {output_dir}")
