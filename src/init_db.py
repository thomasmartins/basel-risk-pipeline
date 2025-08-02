from sqlalchemy import create_engine
from src.models import Base
import os
from dotenv import load_dotenv

load_dotenv()

# Fetch credentials
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')

# Create connection string
connection_string = f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Create engine
engine = create_engine(connection_string)

# Create all tables
Base.metadata.create_all(engine)

print("✅ All tables created successfully.")
