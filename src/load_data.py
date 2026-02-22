import pandas as pd
from sqlalchemy import create_engine


def load_training_data():
    """
    Loads the ML feature dataset from SQL Server into a pandas DataFrame.

    Why this file exists:
    - Centralizes data extraction
    - Makes the dataset reproducible
    - Keeps SQL logic out of modeling code
    """

    # --------------------------------------------------
    # 1️⃣ Create a database connection
    # --------------------------------------------------
    # SQLAlchemy creates a reusable connection engine.
    # The connection string should come from config.py or environment variables.
    engine = create_engine(
        "mssql+pyodbc://sa:Raiders#001!@MARS-Laptop\MARODDB/CMDB_Real_DB?driver=ODBC+Driver+17+for+SQL+Server"
    )

    # --------------------------------------------------
    # 2️⃣ Define the SQL query
    # --------------------------------------------------
    # We SELECT from the VIEW, not the base tables.
    # This guarantees we always pull the same ML-ready dataset.
    query = """
        SELECT *
        FROM dbo.vw_computer_ml_features
    """

    # --------------------------------------------------
    # 3️⃣ Execute query and load into pandas
    # --------------------------------------------------
    # pandas.read_sql executes the SQL and stores the result as a DataFrame.
    df = pd.read_sql(query, engine)

    # --------------------------------------------------
    # 4️⃣ Return the dataset
    # --------------------------------------------------
    return df