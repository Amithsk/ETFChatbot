import os
import json
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments


# Load environment variables from .env file
load_dotenv()

def create_engine():

    user = 'root'
    password = os.getenv('MYSQL_PASSWORD')
    host = 'localhost'
    db = 'etf'
    
    
    if not all([password]):
        raise ValueError("Missing  MYSQL_PASSWORD")
    # Encode special characters in the password
    password = quote_plus(password)

    engine_url = f"mysql+pymysql://{user}:{password}@{host}/{db}"
    engine = sqlalchemy.create_engine(engine_url)
    return engine

def load_data():
    engine = create_engine()
#To retrieve the etf return details
    query = """
        SELECT * 
        FROM etf 
        JOIN etf_returns USING(etf_id)
    """
    df = pd.read_sql(query, engine)
    return df
#Convert DataFrame to Prompt-Response Pairs
def generate_prompt_response_pairs(df):
    pairs = []

    # Pivot ETF returns into columns by time period
    pivot_df = df.pivot_table(
        index=['etf_id', 'etf_name', 'etf_asset_category'],
        columns='etf_returns_timeperiod',
        values='etf_returnsvalue',
        aggfunc='first'
    ).reset_index()

    # 1. Prompt-response per ETF
    for _, row in pivot_df.iterrows():
        name = row['etf_name']
        r1 = row.get('1Y', None)
        r3 = row.get('3Y', None)
        rs = row.get('Since Inception', None)

        if pd.notna(r1):
            pairs.append({
                "prompt": f"What is the 1Y return of {name}?",
                "response": f"The 1-year return of {name} is {r1:.2f}%."
            })

        if pd.notna(r3):
            pairs.append({
                "prompt": f"What is the 3Y return of {name}?",
                "response": f"The 3-year return of {name} is {r3:.2f}%."
            })

        if pd.notna(rs):
            pairs.append({
                "prompt": f"What is the return of {name} since launch?",
                "response": f"The return of {name} since launch is {rs:.2f}%."
            })

    # 2. Highest and Lowest return ETF (1Y only)
    if '1Y' in pivot_df.columns:
        max_row = pivot_df.loc[pivot_df['1Y'].idxmax()]
        min_row = pivot_df.loc[pivot_df['1Y'].idxmin()]

        pairs.append({
            "prompt": "Which ETF has the highest 1Y return?",
            "response": f"The ETF with the highest 1-year return is {max_row['etf_name']} with {max_row['1Y']:.2f}%."
        })

        pairs.append({
            "prompt": "Which ETF has the lowest 1Y return?",
            "response": f"The ETF with the lowest 1-year return is {min_row['etf_name']} with {min_row['1Y']:.2f}%."
        })

    # 3. Average 1Y return by Asset Category
    if '1Y' in pivot_df.columns:
        category_avg = pivot_df.groupby('etf_asset_category')['1Y'].mean().reset_index()
        for _, row in category_avg.iterrows():
            category = row['etf_asset_category']
            avg = row['1Y']
            pairs.append({
                "prompt": f"What is the average 1Y return of {category} ETFs?",
                "response": f"The average 1-year return of ETFs in the {category} category is {avg:.2f}%."
            })

    return pairs

#Format Inputs for Mistral
def format_for_causal_lm(example):
    return {
        "text": f"<s>[Prompt] {example['prompt']} [/Prompt] [Answer] {example['response']} [/Answer]</s>"
    }



if __name__ == "__main__":
    df = load_data()
    print("The columns",df.columns)
    print(df['etf_returns_timeperiod'].unique())
    pairs = generate_prompt_response_pairs(df)
    
    #Save as JSONL (For Fine-Tuning)
    os.makedirs("Code/Training", exist_ok=True)
    with open("Code/Training/etf_qa_dataset.jsonl", "w") as f:
        for pair in pairs:
            json.dump(pair, f)
            f.write("\n")
#Load Dataset with Hugging Face
    dataset = load_dataset("json", data_files="Code/Training/etf_qa_dataset.jsonl", split="train")
 
#To format inputes for mistral
    dataset = dataset.map(format_for_causal_lm)

#Train the model with AutoModelForCausalLM
# Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
#Fix padding token issue
    tokenizer.pad_token = tokenizer.eos_token  
# Initialize model
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", trust_remote_code=True)
    

# Tokenize the dataset
    def tokenize(example):
        encoding = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
             )
        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    tokenized = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir="./etf_mistral_finetuned",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=100,
        save_total_limit=2,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
    )

    try:
        trainer.train()
    #To store the trained model
        os.makedirs("Code/Training/etf_mistral_finetuned", exist_ok=True)
        trainer.save_model("Code/Training/etf_mistral_finetuned")
        tokenizer.save_pretrained("Code/Training/etf_mistral_finetuned")
    except Exception as e:
        print("Training failed:", e)

    