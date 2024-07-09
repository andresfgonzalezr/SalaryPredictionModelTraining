import pandas as pd
from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()


db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)
sql_query= "SELECT * FROM final_data_andres"
df_final1 = pd.read_sql(sql_query, engine)
