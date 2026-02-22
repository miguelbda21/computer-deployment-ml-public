import os
import urllib.parse

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Load .env variables (from project root)
load_dotenv()

SQL_SERVER = (os.getenv("SQL_SERVER") or "").strip()
SQL_DATABASE = (os.getenv("SQL_DATABASE") or "").strip()
SQL_USERNAME = (os.getenv("SQL_USERNAME") or "").strip()
SQL_PASSWORD = (os.getenv("SQL_PASSWORD") or "").strip()
SQL_DRIVER = (os.getenv("SQL_DRIVER") or "ODBC Driver 17 for SQL Server").strip()

SQL_ENCRYPT = (os.getenv("SQL_ENCRYPT") or "no").strip().lower()  # yes/no
SQL_TRUSTED_CONNECTION = (os.getenv("SQL_TRUSTED_CONNECTION") or "no").strip().lower()  # yes/no

_engine: Engine | None = None


def get_engine() -> Engine:
    """
    Returns a singleton SQLAlchemy engine for SQL Server using pyodbc + ODBC Driver.
    Used by pandas.read_sql() and the rest of the pipeline.
    """
    global _engine
    if _engine is not None:
        return _engine

    if not SQL_SERVER or not SQL_DATABASE:
        raise ValueError("Missing SQL_SERVER or SQL_DATABASE in .env")

    # Choose authentication mode
    if SQL_TRUSTED_CONNECTION in ("yes", "true", "1"):
        # Windows Authentication (Integrated Security)
        odbc_conn_str = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            "Trusted_Connection=yes;"
            f"Encrypt={SQL_ENCRYPT};"
            "TrustServerCertificate=yes;"
        )
    else:
        # SQL Authentication (username/password)
        if not SQL_USERNAME or not SQL_PASSWORD:
            raise ValueError("SQL_USERNAME and SQL_PASSWORD are required when SQL_TRUSTED_CONNECTION=no")

        odbc_conn_str = (
            f"DRIVER={{{SQL_DRIVER}}};"
            f"SERVER={SQL_SERVER};"
            f"DATABASE={SQL_DATABASE};"
            f"UID={SQL_USERNAME};"
            f"PWD={SQL_PASSWORD};"
            f"Encrypt={SQL_ENCRYPT};"
            "TrustServerCertificate=yes;"
        )

    encoded = urllib.parse.quote_plus(odbc_conn_str)
    connection_url = f"mssql+pyodbc:///?odbc_connect={encoded}"

    _engine = create_engine(
        connection_url,
        fast_executemany=True,
        pool_pre_ping=True,
    )
    return _engine
