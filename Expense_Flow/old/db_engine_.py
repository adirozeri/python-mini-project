# import os
# import sqlite3
# import pandas as pd
# from datetime import datetime
# import logging

# logger = logging.getLogger('HomeBudget')

# class ExpenseDB:
#     def __init__(self, check_same_thread=False, db_path='expenses.db'):
#         logger.info("initializing db1")
#         self.db_path = db_path
#         self.conn = sqlite3.connect(db_path,check_same_thread=check_same_thread)
#         self.cursor = self.conn.cursor()

#     # def __init__(self, db_path='expenses.db'):
#     def create_and_insert_table(self, df, table_name, if_exists='replace'):
#         try:
#             df.to_sql(
#                 name=table_name,
#                 con=self.conn,
#                 if_exists=if_exists,
#                 index=False
#             )
#             self.conn.commit()
#             logger.info(f"Loaded {len(df)} rows to table '{table_name}'")
            
#         except sqlite3.Error as e:
#             logger.info(f"Error loading data: {e}")
#             self.conn.rollback()
#             raise

#     # def backup_db(self):
#     #    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     #    backup_dir = 'bk_db'
#     #    backup_path = f'{backup_dir}/expenses_{timestamp}.db'
       
#     #    os.makedirs(backup_dir, exist_ok=True)
       
#     #    with sqlite3.connect(self.db_path) as source:
#     #        backup = sqlite3.connect(backup_path)
#     #        source.backup(backup)
#     #        backup.close()
#     #        logger.info(f"Database backed up to {backup_path}")

#     # def restore_from_backup(self, backup_name):
#     #     backup_path = os.path.join('bk_db', backup_name)
#     #     if not os.path.exists(backup_path):
#     #         raise FileNotFoundError(f"Backup not found: {backup_name}")
            
#     #     with sqlite3.connect(backup_path) as backup:
#     #         dest = sqlite3.connect(self.db_path)
#     #         backup.backup(dest)
#     #         dest.close()
#     #         logger.info(f"Database restored from {backup_name}")
            
#     # def list_backups(self):
#     #     backup_dir = 'bk_db'
#     #     if not os.path.exists(backup_dir):
#     #         return []
#     #     return [f for f in os.listdir(backup_dir) if f.endswith('.db')]


#     ########################################################################################################################
#     ########################################################################################################################

#     # def load_business_types(self, df):
#     #     with sqlite3.connect(self.db_path) as conn:
#     #         df.to_sql('business_types', conn, if_exists='replace', index=False)
#     #         logger.info("Business types loaded")

#     # def load_expenses(self, df):
#     #     with sqlite3.connect(self.db_path) as conn:
#     #         df = df.copy()
#     #         # df['charge_date'] = pd.to_datetime(df['charge_date'])
#     #         # df['purchase_date'] = pd.to_datetime(df['purchase_date'])
#     #         df.to_sql('expenses', conn, if_exists='replace', index=False)
#     #         logger.info("Expenses loaded")
    
#     # def get_business_types(self):
#     #     """Retrieve all business types"""
#     #     with sqlite3.connect(self.db_path) as conn:
#     #         return pd.read_sql('SELECT * FROM business_types', conn)

#     # def get_expenses(self, start_date=None, end_date=None):
#     #     """Retrieve expenses with optional date filtering"""
#     #     query = 'SELECT * FROM expenses'
#     #     params = []
        
#     #     if start_date and end_date:
#     #         query += ' WHERE charge_date BETWEEN ? AND ?'
#     #         params = [start_date, end_date]
            
#     #     with sqlite3.connect(self.db_path) as conn:
#     #         df = pd.read_sql(query, conn, params=params)
#     #         df['charge_date'] = pd.to_datetime(df['charge_date'])
#     #         df['purchase_date'] = pd.to_datetime(df['purchase_date'])
#     #         return df
   
#     # def reset_db(self):
#     #     """Drops and recreates all tables"""
#     #     with sqlite3.connect(self.db_path) as conn:
#     #         conn.execute("DROP TABLE IF EXISTS expenses")
#     #         conn.execute("DROP TABLE IF EXISTS business_types")
#     #         self.init_db()
#     #         logger.info("Database reset completed")

#     # def reload_data(self):
#     #     self.backup_db(self)
#     #     self.reset_db(self)
#     #     #populate db
    
