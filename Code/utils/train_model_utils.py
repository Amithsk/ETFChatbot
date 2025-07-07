# ===== train_model_utils.py =====
import os
import datetime
import shutil
from datasets import Dataset

# HuggingFace + PEFT Imports
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# Configuration constants (shared with orchestration file)
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_DIR = os.getcwd()  # ✅ FIXED: No os.pardir
# Paths relative to project root
MODEL_ROOT = os.path.join(BASE_DIR, "Models", "Training", "ModelTraining", RUN_TS)
LOG_ROOT = "logs"

OS_MAKE_DIRS = [MODEL_ROOT, LOG_ROOT]
for _dir in OS_MAKE_DIRS:
    os.makedirs(_dir, exist_ok=True)



def list_checkpoints(output_dir):
    """Return sorted checkpoint directories in `output_dir`."""
    if not os.path.isdir(output_dir):
        return []
    ckpts = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    return sorted(ckpts, key=lambda x: int(x.split("-")[1]))


def get_resume_checkpoint(output_dir):
    """Return path to latest checkpoint, or None if none."""
    ckpts = list_checkpoints(output_dir)
    if not ckpts:
        return None
    return os.path.join(output_dir, ckpts[-1])


def fine_tune_chunk(metric_name: str, prompt_response_pairs: list, chunk_idx: int):
    """
    Fine-tune a LoRA adapter on a chunk of data.
    Resumes from last checkpoint if available.
    """
    # Prepare directories
    adapter_dir = os.path.join(MODEL_ROOT, f"{metric_name}_chunk_{chunk_idx:02d}")
    os.makedirs(adapter_dir, exist_ok=True)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_auth_token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_conf = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_conf)

    # Build dataset
    ds = Dataset.from_list([
        {"text": f"[Prompt] {ex['prompt']} [/Prompt] [Answer] {ex['response']} [/Answer]"}
        for ex in prompt_response_pairs
    ])
    ds = ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        remove_columns=["text"],
    )

    # Training arguments
    output_dir = adapter_dir
    logging_dir = os.path.join(LOG_ROOT, f"{metric_name}_{RUN_TS}")
    os.makedirs(logging_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=20,
        logging_dir=logging_dir,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Resume from checkpoint if available
    resume_ckpt = get_resume_checkpoint(adapter_dir)
    if resume_ckpt:
        print(f"[INFO] Resuming {metric_name} chunk {chunk_idx} from {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print(f"[INFO] Training new {metric_name} chunk {chunk_idx}")
        trainer.train()

    # Save adapter & tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(adapter_dir)
    print(f"[INFO] Saved chunk {chunk_idx} at {adapter_dir}")

    return adapter_dir
