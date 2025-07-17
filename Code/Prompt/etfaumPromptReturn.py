import datetime
import random
import pandas as pd

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
    current_year = df['etf_aum_year'].max()

    # 1. AUM last year
    for _, row in df.iterrows():
        etf = row['etf_name']
        year = int(row['etf_aum_year']) - 1
        value = float(row['etf_aum'])
        response = f"The AUM of {etf} in {year} was ₹{value:,.0f} Cr."
        for prompt in conversational_variants("What was the AUM of {} last year?", etf):
            pairs.append({"prompt": prompt, "response": response})

    # 2. AUM increase year-over-year
    df['etf_aum_year'] = df['etf_aum_year'].astype(int)
    df['etf_aum'] = df['etf_aum'].astype(float)

    aum_pivot = df.pivot_table(
    index='etf_name',
    columns='etf_aum_year',
    values='etf_aum',
    aggfunc='mean'  # or 'sum', 'max', etc., depending on your use case
        )

    for etf in aum_pivot.index:
        years = sorted(aum_pivot.columns)
    for i in range(1, len(years)):
        prev_year = years[i - 1]
        curr_year = years[i]
        prev_value = aum_pivot.loc[etf, prev_year]
        curr_value = aum_pivot.loc[etf, curr_year]

        if pd.notna(prev_value) and pd.notna(curr_value):
            increase = curr_value - prev_value
            response = (
                f"The AUM of {etf} increased by ₹{increase:,.0f} Cr in {curr_year} compared to {prev_year}."
            )
            for prompt in conversational_variants("What was the AUM increase for {}?", etf):
                pairs.append({"prompt": prompt, "response": response})

        if pd.notna(prev_value) and pd.notna(curr_value):
            increase = curr_value - prev_value
            response = (
                f"The AUM of {etf} increased by ₹{increase:,.0f} Cr in {curr_year} compared to {prev_year}."
            )
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