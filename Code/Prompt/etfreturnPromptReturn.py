import os
import pandas as pd
import sqlalchemy
from urllib.parse import quote_plus
from dotenv import load_dotenv

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
    query = """
        SELECT * 
        FROM etf 
        JOIN etf_returns USING(etf_id)
    """
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
