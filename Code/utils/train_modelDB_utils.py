# utils/db_utils.py
import os
import sqlalchemy
from urllib.parse import quote_plus
from dotenv import load_dotenv
import pandas as pd

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

def fetch_etf_returns():
    engine = create_engine()
    query = """
        SELECT 
            e.etf_id, e.etf_name, e.etf_asset_category,
            r.etf_returns_timeperiod, r.etf_returnsvalue,
            r.etf_returnsmonth, r.etf_returnsyear
        FROM etf e
        JOIN etf_returns r USING(etf_id)
    """
    return pd.read_sql(query, engine)

def fetch_etf_expense_ratios():
    engine = create_engine()
    query = """
        SELECT e.etf_id,e.etf_asset_category,e.etf_name,
          r.etf_expenseratio_value,r.etf_expenseratiomonth,r.etf_expenseratioyear
        FROM etf e
        JOIN etf_expenseratio r ON e.etf_id = r.etf_id
    """
    return pd.read_sql(query, engine)

# Later: add tracking error, AUM, etc.
