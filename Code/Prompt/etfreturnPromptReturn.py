import datetime
import random
import pandas as pd

# === Helper: Conversational Variants ===
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

# === Prompt Generator for Returns ===
def generate_prompt_response_return_pairs(df):
    df = df.copy()
    df['return_date'] = pd.to_datetime(
        df['etf_returnsmonth'] + ' ' + df['etf_returnsyear'].astype(str),
        format='%b %Y',
        errors='coerce'
    )

    # Templates rewritten for more natural phrasing
    prompt_templates = {
        "1Y": [
            "What was the return of {} last year?",
            "How did {} perform in the last year?",
            "Give me the 1-year return for {}.",
        ],
        "3Y": [
            "What is the 3-year return of {}?",
            "How much has {} returned over the past 3 years?",
            "Show me how {} performed over 3 years.",
        ],
        "Since Inception": [
            "What is the return of {} since it launched?",
            "How has {} performed since inception?",
            "Show the return since {} was started.",
        ],
    }

    result = []

    for period in ["1Y", "3Y", "Since Inception"]:
        period_df = df[df['etf_returns_timeperiod'] == period]
        latest_df = period_df.sort_values(['etf_id', 'return_date']) \
                             .drop_duplicates('etf_id', keep='last')

        for _, row in latest_df.iterrows():
            name = row['etf_name']
            value = row['etf_returnsvalue']
            r_date = row['return_date'].strftime('%b %Y') if pd.notnull(row['return_date']) else "N/A"
            response = f"The {period} return of {name} as of {r_date} was {value:.2f}%."

            for template in prompt_templates[period]:
                for prompt in conversational_variants(template, name):
                    result.append({"prompt": prompt, "response": response})

    print(f"[INFO] Generated {len(result)} prompt-response pairs")
    return result
