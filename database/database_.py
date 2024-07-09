import pandas as pd
from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()


db_url_1 = os.getenv('DATABASE_URL')
engine_1 = create_engine(db_url_1)
sql_query_1 = "SELECT * FROM final_data_andres"
df_final1 = pd.read_sql(sql_query_1, engine_1)


# Base = declarative_base()

# Base.metadata.create_all(bind=engine_1)
# SessionLocal = sessionmaker(bind=engine_1)
# session = SessionLocal()