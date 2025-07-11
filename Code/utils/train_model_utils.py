# ===== train_model_utils.py =====
import os
import datetime
import shutil
from datasets import Dataset
import tempfile
from peft import PeftModel, PeftConfig

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
from utils.train_modelCheckpoint_utils import configuration_constants, get_global_resume_state

# Configuration constants (shared with orchestration file)
BASE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
RUN_TS, GLOBAL_ROOT, MODEL_ROOT, FINAL_ROOT, LOG_ROOT = configuration_constants()

def fine_tune_chunk(metric_name: str, prompt_response_pairs: list, chunk_idx: int, resume_from_checkpoint: str = None, base_model_path=BASE_MODEL_ID):
    """
    Fine-tune a LoRA adapter on a chunk of data.
    Resumes from last checkpoint if available.
    """
    # Prepare directories
    adapter_dir = os.path.join(MODEL_ROOT, f"{metric_name}_chunk_{chunk_idx:02d}")
    os.makedirs(adapter_dir, exist_ok=True)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_auth_token=HUGGINGFACE_TOKEN if "mistralai" in base_model_path else None)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model (can be original base model or previous chunk's output)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=HUGGINGFACE_TOKEN if "mistralai" in base_model_path else None,
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
        max_steps=50,
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
    if resume_from_checkpoint:
        print(f"[INFO] Resuming {metric_name} chunk {chunk_idx} from {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        print(f"[INFO] Training new {metric_name} chunk {chunk_idx}")
        trainer.train()

    # Save adapter & tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(adapter_dir)
    print(f"[INFO] Saved chunk {chunk_idx} at {adapter_dir}")

    return adapter_dir

def merge_lora_with_base(base_model_id, lora_path, save_path,cleanup_offload=True):
    """
    Merges the LoRA adapter with the base model and saves the merged full model.
    """
    print(f"[INFO] Merging LoRA at {lora_path} with base model {base_model_id}...")
    
   # Use a temporary folder for offloading
    offload_dir = tempfile.mkdtemp(prefix="offload_")

    try:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=False,
        offload_folder=offload_dir,
        offload_state_dict=True  # Important for large model compatibility
     )

        # Load LoRA
        model = PeftModel.from_pretrained(
        base_model,
        model_id=lora_path,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=offload_dir,
        offload_state_dict=True,
        )
        model = model.merge_and_unload()

        # Save full merged model
        model.save_pretrained(save_path)
        model.config.to_json_file(os.path.join(save_path, "config.json"))
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        tokenizer.save_pretrained(save_path)

        print(f"[DONE] Merged model saved at {save_path}")
    finally:
        # Clean up offload folder if requested
        if cleanup_offload and os.path.exists(offload_dir):
            shutil.rmtree(offload_dir)
            print(f"[CLEANUP] Deleted offload cache at {offload_dir}")

