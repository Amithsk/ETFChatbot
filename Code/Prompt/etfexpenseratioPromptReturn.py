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

def generate_expense_ratio_pairs(df):
    pairs = []
    current_year = datetime.datetime.now().year

    # --- 1. Annual Expense Ratio ---
    for _, row in df.iterrows():
        etf, year, value = row['etf_name'], int(row['etf_expenseratioyear']), row['etf_expenseratio_value']
        response = f"The expense ratio of {etf} in {year} was {value:.2f}%."
        for prompt in conversational_variants("What was the expense ratio of {} in {}?", etf, year):
            pairs.append({"prompt": prompt, "response": response})
        for prompt in conversational_variants("Tell me the expense ratio for {} for the year {}.", etf, year):
            pairs.append({"prompt": prompt, "response": response})

    # --- 2. Multi-Year Trend ---
    for etf, group in df.groupby("etf_name"):
        for n in [3, 5]:
            years = [current_year - i for i in range(n)][::-1]
            values = group[group['etf_expenseratioyear'].isin(years)].sort_values('etf_expenseratioyear')
            if len(values) == n:
                trend = ", ".join(f"{int(row.year)}: {row.expense_ratio:.2f}%" for _, row in values.iterrows())
                response = f"The expense ratio trend of {etf} over the last {n} years is: {trend}."
                for prompt in conversational_variants("How has the expense ratio of {} changed over the last {} years?", etf, n):
                    pairs.append({"prompt": prompt, "response": response})
                for prompt in conversational_variants("Show me the trend in expense ratio for {} in recent years.", etf):
                    pairs.append({"prompt": prompt, "response": response})

    # --- 3. Min/Max Expense Ratio in a Year ---
    for year in df['etf_expenseratioyear'].unique():
        data = df[df['etf_expenseratioyear'] == year]
        if not data.empty:
            min_row = data.loc[data['etf_expenseratio_value'].idxmin()]
            max_row = data.loc[data['etf_expenseratio_value'].idxmax()]
            pairs.append({
                "prompt": f"Which ETF had the lowest expense ratio in {int(year)}?",
                "response": f"The ETF with the lowest expense ratio in {int(year)} was {min_row['etf_name']} with {min_row['etf_expenseratio_value']:.2f}%."
            })
            pairs.append({
                "prompt": f"Which ETF had the highest expense ratio in {int(year)}?",
                "response": f"The ETF with the highest expense ratio in {int(year)} was {max_row['etf_name']} with {max_row['etf_expenseratio_value']:.2f}%."
            })

    # --- 4. Min/Max Average Over Last N Years ---
    for n in [3, 5]:
        years = [current_year - i for i in range(n)]
        subset = df[df['etf_expenseratioyear'].isin(years)]
        avg_expenses = subset.groupby('etf_name')['etf_expenseratio_value'].mean()
        if not avg_expenses.empty:
            min_etf, min_val = avg_expenses.idxmin(), avg_expenses.min()
            max_etf, max_val = avg_expenses.idxmax(), avg_expenses.max()
            pairs.append({
                "prompt": f"What ETF had the lowest expense ratio over the last {n} years?",
                "response": f"The ETF with the lowest average expense ratio over the last {n} years was {min_etf} with {min_val:.2f}%."
            })
            pairs.append({
                "prompt": f"What ETF had the highest expense ratio over the last {n} years?",
                "response": f"The ETF with the highest average expense ratio over the last {n} years was {max_etf} with {max_val:.2f}%."
            })

    return pairs
