from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch credentials
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create connection string
connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Connect to database
engine = create_engine(connection_string)

# Example query
df = pd.read_sql('SELECT * FROM lcr_data;', engine)
print(df)

st.title("Basel III Risk Data Dashboard")

# Sidebar selection
metric = st.sidebar.selectbox(
    "Select a Metric",
    ("LCR", "NSFR", "IRRBB Exposures", "RWA Breakdown")
)

# Query functions
def load_lcr():
    query = "SELECT date, lcr_value FROM lcr_data ORDER BY date;"
    return pd.read_sql(query, engine)

def load_nsfr():
    query = "SELECT date, nsfr_value FROM nsfr_data ORDER BY date;"
    return pd.read_sql(query, engine)

def load_irrbb():
    query = """
    SELECT maturity_bucket, SUM(amount) AS total_amount
    FROM irrbb_exposures
    GROUP BY maturity_bucket;
    """
    return pd.read_sql(query, engine)

def load_rwa():
    query = """
    SELECT exposure_class, SUM(ead) AS total_ead, SUM(rwa) AS total_rwa
    FROM rwa_data
    GROUP BY exposure_class;
    """
    return pd.read_sql(query, engine)

# Load and display data
if metric == "LCR":
    df = load_lcr()
    st.subheader("Liquidity Coverage Ratio (LCR)")
    st.dataframe(df)

elif metric == "NSFR":
    df = load_nsfr()
    st.subheader("Net Stable Funding Ratio (NSFR)")
    st.dataframe(df)

elif metric == "IRRBB Exposures":
    df = load_irrbb()
    st.subheader("IRRBB Exposure Breakdown")
    st.dataframe(df)

elif metric == "RWA Breakdown":
    df = load_rwa()
    st.subheader("Risk-Weighted Assets (RWA) Breakdown")
    st.dataframe(df)
