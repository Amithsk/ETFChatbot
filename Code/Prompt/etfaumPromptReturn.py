import datetime
import random

def conversational_variants(template, *args):
    base = template.format(*args)
    variants = [
        base,
        f"Can you tell me {base[0].lower() + base[1:]}?",
        f"Tell me {base[0].lower() + base[1:]}",
        f"Do you know {base.lower()}?",
        f"I'm curious, {base.lower()}",
    ]
    return random.sample(variants, 2)

def generate_aum_pairs(df):
    pairs = []
    current_year = df['aum_year'].max()

    # 1. AUM last year
    for _, row in df.iterrows():
        etf = row['etf_name']
        year = row['aum_year'] - 1
        value = row['aum_prev_year_value']
        response = f"The AUM of {etf} in {year} was ₹{value:,.0f} Cr."
        for prompt in conversational_variants("What was the AUM of {} last year?", etf):
            pairs.append({"prompt": prompt, "response": response})

    # 2. AUM increase
    for _, row in df.iterrows():
        etf = row['etf_name']
        increase = row['aum_value'] - row['aum_prev_year_value']
        response = f"The AUM of {etf} increased by ₹{increase:,.0f} Cr in {row['aum_year']} compared to {row['aum_year']-1}."
        for prompt in conversational_variants("What was the AUM increase for {}?", etf):
            pairs.append({"prompt": prompt, "response": response})

    # 3. ETF with highest AUM in gold asset class
    gold_df = df[df['asset_class'].str.lower() == 'gold']
    if not gold_df.empty:
        top_etf = gold_df.loc[gold_df['aum_value'].idxmax()]
        response = f"The gold ETF with the highest AUM is {top_etf['etf_name']} with ₹{top_etf['aum_value']:,} Cr."
        for prompt in conversational_variants("Which gold ETF has the highest AUM?", ""):
            pairs.append({"prompt": prompt.strip(), "response": response})
        for prompt in conversational_variants("Tell me the top AUM ETF in the gold asset class", ""):
            pairs.append({"prompt": prompt.strip(), "response": response})

    return pairs