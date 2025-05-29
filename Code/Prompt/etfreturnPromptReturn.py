import os
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

# Load environment variables from .env file
load_dotenv()

def create_engine():
    user = 'root'
    password = os.getenv('MYSQL_PASSWORD')
    host = 'localhost'
    db = 'etf'
    if not password:
        raise ValueError("Missing MYSQL_PASSWORD")
    password = quote_plus(password)
    engine_url = f"mysql+pymysql://{user}:{password}@{host}/{db}"
    return sqlalchemy.create_engine(engine_url)

def load_data():
    engine = create_engine()
    query = """
        SELECT * 
        FROM etf 
        JOIN etf_returns USING(etf_id)
    """
    return pd.read_sql(query, engine)

def generate_prompt_response_pairs(df):
    pairs = []
    pivot_df = df.pivot_table(
        index=['etf_id', 'etf_name', 'etf_asset_category'],
        columns='etf_returns_timeperiod',
        values='etf_returnsvalue',
        aggfunc='first'
    ).reset_index()

    for _, row in pivot_df.iterrows():
        name = row['etf_name']
        r1, r3, rs = row.get('1Y'), row.get('3Y'), row.get('Since Inception')
        if pd.notna(r1):
            pairs.append({"prompt": f"What is the 1Y return of {name}?", "response": f"The 1-year return of {name} is {r1:.2f}%."})
        if pd.notna(r3):
            pairs.append({"prompt": f"What is the 3Y return of {name}?", "response": f"The 3-year return of {name} is {r3:.2f}%."})
        if pd.notna(rs):
            pairs.append({"prompt": f"What is the return of {name} since launch?", "response": f"The return of {name} since launch is {rs:.2f}%."})

    if '1Y' in pivot_df.columns:
        max_row = pivot_df.loc[pivot_df['1Y'].idxmax()]
        min_row = pivot_df.loc[pivot_df['1Y'].idxmin()]
        pairs.append({"prompt": "Which ETF has the highest 1Y return?", "response": f"The ETF with the highest 1-year return is {max_row['etf_name']} with {max_row['1Y']:.2f}%."})
        pairs.append({"prompt": "Which ETF has the lowest 1Y return?", "response": f"The ETF with the lowest 1-year return is {min_row['etf_name']} with {min_row['1Y']:.2f}%."})

        category_avg = pivot_df.groupby('etf_asset_category')['1Y'].mean().reset_index()
        for _, row in category_avg.iterrows():
            pairs.append({"prompt": f"What is the average 1Y return of {row['etf_asset_category']} ETFs?", "response": f"The average 1-year return of ETFs in the {row['etf_asset_category']} category is {row['1Y']:.2f}%."})

    return pairs

def format_for_causal_lm(example):
    return {"text": f"<s>[Prompt] {example['prompt']} [/Prompt] [Answer] {example['response']} [/Answer]</s>"}

if __name__ == "__main__":
    df = load_data()
    pairs = generate_prompt_response_pairs(df)

    dataset = Dataset.from_list(pairs)
    dataset = dataset.map(format_for_causal_lm)

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)

    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="Code/Training/etf_tinyllama_finetuned",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        fp16=False,  # Not supported on CPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("Number of training samples:", len(tokenized))

    model.save_pretrained("Code/Training/etf_tinyllama_finetuned")
    tokenizer.save_pretrained("Code/Training/etf_tinyllama_finetuned")
