import pandas as pd
from dotenv import load_dotenv

from src.db_connection import get_engine


def test_db_connection_and_query():
    load_dotenv()  # loads .env from project root

    engine = get_engine()

    df = pd.read_sql(
        "SELECT TOP (5) name FROM sys.tables ORDER BY name;",
        engine
    )

    assert not df.empty
    print(df)