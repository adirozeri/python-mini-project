import os
import sqlite3
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger('HomeBudget')


# logger.info(f"Loading {len(df)} records...")



class ExpenseDB:
    def __init__(self, db_path='expenses.db'):
        logger.info("initializing db")
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS business_types (
                business_name TEXT PRIMARY KEY,
                business_type TEXT,
                business_utility TEXT,
                super_name TEXT
            )''')
            
            conn.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY,
                card_id TEXT,
                charge_date DATE,
                purchase_date DATE,
                business_name TEXT,
                charge_nis REAL,
                business_type TEXT,
                business_utility TEXT,
                super_name TEXT,
                charge_days_in_month INTEGER,
                charge_Year INTEGER,
                charge_Day INTEGER,
                charge_Dayofweek INTEGER,
                charge_Dayofyear INTEGER,
                charge_Month INTEGER,
                charge_YearMonth TEXT,
                purchase_YearMonth TEXT,
                days_in_month INTEGER,
                plt_business_name TEXT,
                charge_nis_day_average REAL,
                FOREIGN KEY (business_name) REFERENCES business_types(business_name)
                )'''
                    )

    def load_business_types(self, df):
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql('business_types', conn, if_exists='replace', index=False)
            logger.info("Business types loaded")

    def load_expenses(self, df):
        with sqlite3.connect(self.db_path) as conn:
            df = df.copy()
            # df['charge_date'] = pd.to_datetime(df['charge_date'])
            # df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            df.to_sql('expenses', conn, if_exists='replace', index=False)
            logger.info("Expenses loaded")
    
    def get_business_types(self):
        """Retrieve all business types"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql('SELECT * FROM business_types', conn)

    def get_expenses(self, start_date=None, end_date=None):
        """Retrieve expenses with optional date filtering"""
        query = 'SELECT * FROM expenses'
        params = []
        
        if start_date and end_date:
            query += ' WHERE charge_date BETWEEN ? AND ?'
            params = [start_date, end_date]
            
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(query, conn, params=params)
            df['charge_date'] = pd.to_datetime(df['charge_date'])
            df['purchase_date'] = pd.to_datetime(df['purchase_date'])
            return df
   
    def reset_db(self):
        """Drops and recreates all tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS expenses")
            conn.execute("DROP TABLE IF EXISTS business_types")
            self.init_db()
            logger.info("Database reset completed")

    def reload_data(self):
        self.backup_db(self)
        self.reset_db(self)
        #populate db
    
    def backup_db(self):
       timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
       backup_dir = 'bk_db'
       backup_path = f'{backup_dir}/expenses_{timestamp}.db'
       
       os.makedirs(backup_dir, exist_ok=True)
       
       with sqlite3.connect(self.db_path) as source:
           backup = sqlite3.connect(backup_path)
           source.backup(backup)
           backup.close()
           logger.info(f"Database backed up to {backup_path}")

    def restore_from_backup(self, backup_name):
        backup_path = os.path.join('bk_db', backup_name)
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_name}")
            
        with sqlite3.connect(backup_path) as backup:
            dest = sqlite3.connect(self.db_path)
            backup.backup(dest)
            dest.close()
            logger.info(f"Database restored from {backup_name}")
            
    def list_backups(self):
        backup_dir = 'bk_db'
        if not os.path.exists(backup_dir):
            return []
        return [f for f in os.listdir(backup_dir) if f.endswith('.db')]


# Load data
# biz_types_df = pd.read_csv('biz_types.csv')
# expenses_df = pd.read_csv('main_df.csv')

# Initialize and load
# db = ExpenseDB()
# db.load_business_types(biz_types_df)
# db.load_expenses(expenses_df)