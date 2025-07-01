import os
import datetime
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def train_metric_adapter(metric_name, prompt_response_pairs, base_model_id):
    token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list([
        {
            "text": f"[Prompt] {ex['prompt']} [/Prompt] [Answer] {ex['response']} [/Answer]"
        }
        for ex in prompt_response_pairs
    ])

    dataset = dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
        remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"Models/Training/{metric_name}_LoRA_{timestamp}"
    logging_dir = f"logs/{metric_name}_{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        save_steps=200,
        save_total_limit=2,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=20,
        logging_dir=logging_dir,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print(f"[INFO] Starting training for: {metric_name}")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Training complete for {metric_name}. Saved to {output_dir}")
