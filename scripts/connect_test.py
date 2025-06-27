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

# List of tables to pull
tables = ['lcr_data', 'nsfr_data', 'irrbb_exposures', 'rwa_data']

# Query each table
for table in tables:
    query = f'SELECT * FROM {table};'
    df = pd.read_sql(query, engine)
    print(f"\nTable: {table}")
    print(df)
