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
#Retrieve the ETF and Asset category details from the DB
    query = """
        SELECT e.*, r.*, a.asset_info
        FROM etf e
        JOIN etf_returns r ON e.etf_id = r.etf_id
        LEFT JOIN etf_asset a ON e.etf_asset_category = a.idetf_asset
    """
    return pd.read_sql(query, engine)

def generate_prompt_response_pairs(df):
    pairs = []

    df = df.loc[:, ~df.columns.duplicated()]
    print(df.columns.tolist())
    df['category'] = df['asset_info'].fillna(df['etf_asset_category'])

    pivot_df = df.pivot_table(
        index=['etf_id', 'etf_name', 'category'],
        columns='etf_returns_timeperiod',
        values='etf_returnsvalue',
        aggfunc='first'
    ).reset_index()
    print("Pivoted columns:", pivot_df.columns.tolist())

    print(df['etf_returns_timeperiod'].unique())

    for _, row in pivot_df.iterrows():
        name = row['etf_name']
        category = row['category']
        for period, label in [('1Y', '1-year'), ('3Y', '3-year'), ('5Y', '5-year'), ('10Y', '10-year'), ('SL', 'since launch')]:
            value = row.get(period)
            if pd.notna(value):
                pairs.append({
                    "prompt": f"What is the {label} return of {name}?",
                    "response": f"The {label} return of {name} is {value:.2f}%."
                })

    # Average return by category and highest-returning category
    for period, label in [('1Y', '1 year'), ('3Y', '3 years'), ('5Y', '5 years'), ('10Y', '10 years')]:
        if period in pivot_df.columns:
            category_avg = pivot_df.groupby('category')[period].mean().reset_index()
            for _, row in category_avg.iterrows():
                pairs.append({
                    "prompt": f"What is the average {period} return of {row['category']} ETFs?",
                    "response": f"The average {label} return of ETFs in the {row['category']} category is {row[period]:.2f}%."
                })

            top_cat = category_avg.loc[category_avg[period].idxmax()]
            pairs.append({
                "prompt": f"Which category has the highest return in the last {label}?",
                "response": f"The category with the highest average return in the last {label} is {top_cat['category']} with {top_cat[period]:.2f}%."
            })

    # Highest return since launch within each category
    if 'SL' in pivot_df.columns:
        for category, group in pivot_df.groupby('category'):
            group = group.dropna(subset=['SL'])
            if not group.empty:
                max_row = group.loc[group['SL'].idxmax()]
                pairs.append({
                    "prompt": f"Which ETF has the highest return in the {category} category since launch?",
                    "response": f"The ETF with the highest return in the {category} category since launch is {max_row['etf_name']} with {max_row['SL']:.2f}%."
                })

        # Overall highest and lowest
        max_row = pivot_df.loc[pivot_df['SL'].idxmax()]
        min_row = pivot_df.loc[pivot_df['SL'].idxmin()]
        pairs.append({"prompt": "Which ETF has the highest return since inception ?", "response": f"The ETF with the highest return since inception is {max_row['etf_name']} with {max_row['SL']:.2f}%."})
        pairs.append({"prompt": "Which ETF has the lowest return since inception?", "response": f"The ETF with the lowest return since inception is {min_row['etf_name']} with {min_row['SL']:.2f}%."})

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
        output_dir="Models/Training/etf_tinyllama_finetuned/",
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

    model.save_pretrained("Models/Training/etf_tinyllama_finetuned/")
    tokenizer.save_pretrained("Models/Training/etf_tinyllama_finetuned/")

