import os
import torch
import datetime
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus
from datasets import Dataset
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

load_dotenv()

# === DB Connection ===
def create_engine():
    user = 'root'
    password = os.getenv('MYSQL_PASSWORD')
    host = 'localhost'
    db = 'etf'
    if not password:
        raise ValueError("Missing MYSQL_PASSWORD in .env")
    password = quote_plus(password)
    engine_url = f"mysql+pymysql://{user}:{password}@{host}/{db}"
    return sqlalchemy.create_engine(engine_url)

def load_data():
    engine = create_engine()
    query = """
        SELECT 
            e.etf_id, e.etf_name, e.etf_asset_category,
            r.etf_returns_timeperiod, r.etf_returnsvalue,
            r.etf_returnsmonth, r.etf_returnsyear
        FROM etf e
        JOIN etf_returns r USING(etf_id)
    """
    df = pd.read_sql(query, engine)
    df['etf_returnsmonth'] = df['etf_returnsmonth'].str.title().str[:3]
    return df

# === Prompt Generator ===
def generate_prompt_response_pairs(df):
    today = datetime.date.today()
    ref_month = today.strftime('%b')       # e.g. "Jun"
    ref_year_1y = today.year - 1
    ref_year_3y = today.year - 3

    df_1y = df[
        (df['etf_returns_timeperiod'] == '1Y') &
        (df['etf_returnsmonth'] == ref_month) &
        (df['etf_returnsyear'] == ref_year_1y)
    ]

    df_3y = df[
        (df['etf_returns_timeperiod'] == '3Y') &
        (df['etf_returnsmonth'] == ref_month) &
        (df['etf_returnsyear'] == ref_year_3y)
    ]

    df_si = df[df['etf_returns_timeperiod'] == 'Since Inception']
    df_si = df_si.sort_values(['etf_id', 'etf_returnsyear', 'etf_returnsmonth'])
    df_si = df_si.drop_duplicates('etf_id', keep='last')

    prompt_variants_1y = [
        "What is the 1Y return of {name}?",
        "How much did {name} return last year?",
        "What was {name}'s return over the past year?",
        "Tell me the one-year return for {name}?",
        "Last year return of {name}?"
    ]
    prompt_variants_3y = [
        "What is the 3Y return of {name}?",
        "How much has {name} returned in 3 years?",
        "What was the 3-year return for {name}?",
        "Return of {name} over the past 3 years?"
    ]
    prompt_variants_si = [
        "What is the return of {name} since launch?",
        "How much has {name} returned since inception?",
        "What is the return since inception for {name}?",
        "Total return of {name} from launch date?"
    ]

    pairs = []

    for _, row in df_1y.iterrows():
        name, value = row['etf_name'], row['etf_returnsvalue']
        for template in prompt_variants_1y[:2]:
            prompt = template.format(name=name)
            response = f"The 1-year return of {name} as of {ref_month} {ref_year_1y} was {value:.2f}%."
            pairs.append({"prompt": prompt, "response": response})

    for _, row in df_3y.iterrows():
        name, value = row['etf_name'], row['etf_returnsvalue']
        for template in prompt_variants_3y[:2]:
            prompt = template.format(name=name)
            response = f"The 3-year return of {name} as of {ref_month} {ref_year_3y} was {value:.2f}%."
            pairs.append({"prompt": prompt, "response": response})

    for _, row in df_si.iterrows():
        name, value = row['etf_name'], row['etf_returnsvalue']
        for template in prompt_variants_si[:2]:
            prompt = template.format(name=name)
            response = f"The return of {name} since inception is {value:.2f}%."
            pairs.append({"prompt": prompt, "response": response})

    print(f"[INFO] Generated {len(pairs)} prompt-response pairs")
    return pairs

# === Main Training Function ===
def main():
    df = load_data()
    pairs = generate_prompt_response_pairs(df)
    dataset = Dataset.from_list(pairs)
    dataset = dataset.map(lambda ex: {
        "text": f"<s>[Prompt] {ex['prompt']} [/Prompt] [Answer] {ex['response']} [/Answer]</s>"
    })

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    token = os.getenv("HUGGINGFACE_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_batch, batched=True, remove_columns=["text", "prompt", "response"])

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        token=token
    )
    model = prepare_model_for_kbit_training(model)

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

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"Models/Training/Mistral-LoRA-{timestamp}"
    logging_dir = f"logs/MistralETF-{timestamp}"

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
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("[INFO] Starting training...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[INFO] Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main()
