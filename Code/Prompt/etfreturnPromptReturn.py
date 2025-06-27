import os
import datetime
import pandas as pd
import sqlalchemy
from collections import Counter
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datasets import Dataset
import datetime
import random
import torch
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
    df = pd.read_sql(query, engine)
    print(f"[INFO] Retrieved {len(df)} rows from the database.")
    #csv_file_path = f"Models/Training/etf_data_export.csv"
    #df.to_csv(csv_file_path, index=False)
    #print(f"[INFO] Data written to {csv_file_path}")

    
    return pd.read_sql(query, engine)




def conversational_variants(template, *args):
    """Generate conversational prompt variants."""
    base = template.format(*args)
    variants = [
        base,
        f"Can you tell me {base[0].lower() + base[1:]}",
        f"I'm curious, {base.lower()}",
        f"Do you know {base.lower()}",
        f"Would you happen to know {base.lower()}",
    ]
    return random.sample(variants, 2)  # Pick 2 variants randomly for variety

def generate_prompt_response_etfreturn_pairs(df):
    pairs = []
    seen_prompts = set()
    seen_facts = set()
    df = df.loc[:, ~df.columns.duplicated()]
    df['category'] = df['asset_info'].where(df['asset_info'].notna() & (df['asset_info'] != df['etf_name']), df['etf_asset_category'])
    current_year = datetime.datetime.now().year

    df_filtered = df.dropna(subset=['etf_returnsvalue', 'etf_returnsyear', 'etf_name']).copy()
    df_filtered['etf_returnsyear'] = pd.to_numeric(df_filtered['etf_returnsyear'], errors='coerce')
   

    annual_returns = (
        df_filtered.groupby(['etf_name', 'etf_returnsyear'])['etf_returnsvalue']
        .first()
        .unstack(level=1)
    )

    def add_pair(prompt, response):
        p_clean = prompt.strip().lower()
        if p_clean not in seen_prompts and response not in seen_facts:
            seen_prompts.add(p_clean)
            seen_facts.add(response)
            pairs.append({"prompt": prompt, "response": response})

    # --- ETF-specific Returns by Year ---
    for etf in annual_returns.index:
        for year in annual_returns.columns:
            value = annual_returns.loc[etf, year]
            if pd.notna(value):
                response = f"The return of {etf} in {int(year)} was {value:.2f}%."
                prompt = conversational_variants("What was the return of {} in {}?", etf, int(year))[0]
                add_pair(prompt, response)

    # --- Category Average Return by Year ---
    
    for year in df['etf_returnsyear'].dropna().unique():
        grouped = df[df['etf_returnsyear'] == year].groupby('category')['etf_returnsvalue'].mean().reset_index()
        for _, row in grouped.iterrows():
            response = f"The average return of ETFs in the {row['category']} category in {int(year)} was {row['etf_returnsvalue']:.2f}%."
            prompt = conversational_variants("What was the average return of ETFs in {} category in {}?", row['category'], int(year))[0]
            add_pair(prompt, response)

    # --- Best Avg Category in Last N Years ---
    for n in [2, 3, 5]:
        start_year = current_year - n
        category_returns = df_filtered[df_filtered['etf_returnsyear'].between(start_year, current_year - 1)]
        grouped = category_returns.groupby(['category', 'etf_returnsyear'])['etf_returnsvalue'].mean().unstack()

        if grouped.shape[1] == n:
            for cat in grouped.index:
                values = grouped.loc[cat].dropna()
                if len(values) == n:
                    compounded =round(((values/ 100 + 1).prod() - 1) * 100,2)
                    grouped.loc[cat, 'compounded'] = compounded
            if 'compounded' in grouped.columns:
                top_cat = grouped['compounded'].idxmax()
                top_value = grouped['compounded'].max()
                response = f"The category with the highest average return in the last {n} years is {top_cat} with {top_value:.2f}%."
                prompt = conversational_variants("Which category has the best return in the last {} years?", n)[0]
                add_pair(prompt, response)

    # --- Last N Years ETF Return ---
    for n in [2, 3, 5]:
        start_year = current_year - n
        for etf in annual_returns.index:
            returns = annual_returns.loc[etf, [y for y in range(start_year, current_year) if y in annual_returns.columns]].dropna()
            if len(returns) == n:
                compounded = round(((returns / 100 + 1).prod() - 1) * 100,2)
                response = f"The return of {etf} in the last {n} years was {compounded:.2f}%."
                prompt = conversational_variants("What was the return of {} in the last {} years?", etf, n)[0]
                add_pair(prompt, response)

    # --- Return from Year X to Y ---
    for etf in annual_returns.index:
        years = sorted(annual_returns.columns.dropna())
        for i in range(len(years)):
            for j in range(i + 1, len(years)):
                start = years[i]
                end = years[j]
                returns = annual_returns.loc[etf, [y for y in range(start, end + 1)]].dropna()
                if len(returns) == (end - start + 1):
                    compounded = round(((returns / 100 + 1).prod() - 1) * 100,2)
                    response = f"The return of {etf} from {start} to {end} was {compounded:.2f}%."
                    prompt = conversational_variants("What was the return of {} from {} to {}?", etf, start, end)[0]
                    add_pair(prompt, response)

    # --- Computed Since Launch Returns ---
    computed_sl_returns = []

    for etf in annual_returns.index:
        returns = annual_returns.loc[etf].dropna()
        if len(returns) >= 2:
            start_year = returns.index.min()
            end_year = returns.index.max()
            year_range = list(range(start_year, end_year + 1))
            valid_returns = annual_returns.loc[etf, year_range].dropna()
            if len(valid_returns) == len(year_range):
                compounded = round(((valid_returns / 100 + 1).prod() - 1) * 100,2)
                computed_sl_returns.append((etf, compounded, start_year, end_year))
                response = f"The return of {etf} since launch (from {start_year} to {end_year}) is {compounded:.2f}%."
                prompt = conversational_variants("What is the return of {} since launch?", etf)[0]
                add_pair(prompt, response)

    # --- Best/Worst ETF Since Launch ---
    if computed_sl_returns:
        top = max(computed_sl_returns, key=lambda x: x[1])
        bottom = min(computed_sl_returns, key=lambda x: x[1])

        add_pair(
            "Which ETF has the highest return since inception?",
            f"The ETF with the highest return since launch is {top[0]} with {top[1]:.2f}% (from {top[2]} to {top[3]})."
        )
        add_pair(
            "Which ETF has the lowest return since inception?",
            f"The ETF with the lowest return since launch is {bottom[0]} with {bottom[1]:.2f}% (from {bottom[2]} to {bottom[3]})."
        )

        
    # --- Return of ETF as of specific year (i.e. since launch to given year) ---
    for etf in annual_returns.index:
        returns = annual_returns.loc[etf].dropna()
        if len(returns) >= 2:
            launch_year = returns.index.min()
            max_year = returns.index.max()
            for as_of_year in range(launch_year + 1, max_year + 1):
                year_range = list(range(launch_year, as_of_year + 1))
                available = annual_returns.loc[etf, year_range].dropna()
                if len(available) == len(year_range):
                    compounded = round(((available / 100 + 1).prod() - 1) * 100,2)
                    response = f"The return of {etf} as of {as_of_year} (from {launch_year} to {as_of_year}) is {compounded:.2f}%."
                    prompt = conversational_variants("What is the return of {} as of {}?", etf, as_of_year)[0]
                    add_pair(prompt, response)

    return pairs



#Pass tokenizer explicitly to the function
def format_for_causal_lm(example,tokenizer):
    #This will makes sure the model sees <EOS> as the signal “that’s your entire response.”
    text  = (
        f"<s>[Prompt] {example['prompt']} [/Prompt] "
        f"[Answer] {example['response']} [/Answer] "
        f"{tokenizer.eos_token}"
        f"</s>"
    )
    return {"text": text}

if __name__ == "__main__":
    df = load_data()
    pairs = generate_prompt_response_etfreturn_pairs(df)
    print(f"Generated pairs: {len(pairs)}")
    responses = [p['response'] for p in pairs]
    print(f"Unique responses: {len(set(responses))}")
    

    dataset = Dataset.from_list(pairs)
    

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #To ensure the model stop at the EOS token
    tokenizer.add_special_tokens({
    "eos_token": "<EOS>",
    "additional_special_tokens": ["<EOS>"]
        })
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda x: format_for_causal_lm(x, tokenizer))

    model = AutoModelForCausalLM.from_pretrained(model_id)
    #To resize the model
    model.resize_token_embeddings(len(tokenizer))
    if torch.cuda.is_available():
        model = model.to("cuda")

    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    training_args = TrainingArguments(
        output_dir=f"Models/Training/etf_tinyllama_finetuned_return_{timestamp}/",
        per_device_train_batch_size=2,
        num_train_epochs=1,# (optional) disable epoch counting so you don’t trigger warnings:
        max_steps=100,#To limit the training,post validation comment this one training is done
        save_steps=100,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,  # only works with NVIDIA GPU
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

    model.save_pretrained(f"Models/Training/etf_tinyllama_finetuned_return_{timestamp}/")
    tokenizer.save_pretrained(f"Models/Training/etf_tinyllama_finetuned_return_{timestamp}/")

