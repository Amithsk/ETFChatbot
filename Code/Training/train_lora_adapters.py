from Prompt.etfexpenseratioPromptReturn import generate_expense_ratio_pairs
from Prompt.etfreturnPromptReturn import generate_prompt_response_return_pairs
from utils.train_model_utils import train_metric_adapter
from utils.train_modelDB_utils import fetch_etf_expense_ratios,fetch_etf_returns


df_returns = fetch_etf_returns()
pairs = generate_expense_ratio_pairs(df_returns)
train_metric_adapter("expense_ratio", pairs)

df_expense  = fetch_etf_expense_ratios()
pairs = generate_prompt_response_return_pairs(df_expense)
train_metric_adapter("return_ratio",pairs)

