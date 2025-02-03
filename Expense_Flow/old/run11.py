
import streamlit as st
st.set_page_config(
    page_title="HomeBudget",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import logic1
import db_engine1
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import plots
print("Script executing", datetime.now())
requirements = [
    'streamlit',
    'pandas',
    'matplotlib',
    'seaborn',
    'numpy',
    'python-dateutil'
]


# def initialize_database(logger):
#     """Initialize database and load data if needed. This should only run once."""
#     args = parse_args()
#     db = db_engine.ExpenseDB()
#     logger.info('initializing db - done')
    
#     if args.loaddata:
#         logger.info('getting data')
#         df, biz_types_df = logic1.load_data()
#         logger.info('getting data - done')
#         db.load_business_types(biz_types_df)
#         db.load_expenses(df)
    
#     return db



def setup_logging():
    """Configure rotating logging for the HomeBudget application"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure base logger
    logger = logging.getLogger('HomeBudget')

    if logger.hasHandlers():
        logger.handlers.clear()


    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create rotating file handler
    log_file = log_dir / "homebudget.log"
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5*1024*1024,  # 5MB per file
        backupCount=5,         # Keep 5 backup files
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging system initialized")
    return logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loaddata', action='store_true', help='Reload data into database')
    return parser.parse_args()



# def main(db):

    # args = parse_args()
        
    # if args.loaddata:
    #     logger.info('Load data - load_data')
    #     expenses_df, business_df = logic1.load_data()
    #     logger.info('Load data - expenses_df, business_df created')

    #     db.create_and_insert_table(expenses_df, 'expenses')
    #     logger.info('Load data - expenses_df done')

    #     db.create_and_insert_table(business_df,'business_types')
    #     logger.info('Load data - business_df done')


        
    


@st.cache_resource
def get_database():
    """This will persist even after page refresh"""
    db = db_engine1.ExpenseDB(check_same_thread=False)
    return db

@st.cache_data
def load_data():
    """Cache the data loading separately"""
    return logic1.load_data()

@st.cache_data
def initialize_db(_db):
    """Cache the data loading"""
    # expenses_df, business_df = logic1.load_data()
    # db.create_and_insert_table(expenses_df, 'expenses')
    # db.create_and_insert_table(business_df,'business_types')

    logger.info('start create data - load_data')

    expenses_df, business_df = load_data()

    
    logger.info('done create data - expenses_df, business_df created')
    logger.info('start create data - expenses_df')

    _db.create_and_insert_table(expenses_df, 'expenses')

    logger.info('Load data - expenses_df done')
    logger.info('start create data - business_df')

    _db.create_and_insert_table(business_df,'business_types')

    logger.info('Load data - business_df done')
    


    return True

def mid_calc(db):
        df = pd.read_sql("SELECT * FROM expenses", db.conn)

        today = datetime.today()
        current_month = today.month
        current_YearMonth = datetime.now().strftime('%m-%Y')
        last_month_total_days = (datetime.today() - relativedelta(months=1)).replace(day=15)
        last_month_days_difference = (today - last_month_total_days).days

        total_days_df = df[['charge_YearMonth','charge_days_in_month']].drop_duplicates()
        total_days_without_last_month = total_days_df[total_days_df['charge_YearMonth']!=current_YearMonth]['charge_days_in_month'].sum()
        total_months = (total_days_without_last_month + last_month_days_difference)/(365/12)
        figsize=(5,3)

        means = df.groupby('charge_YearMonth')['charge_nis_day_average'].sum()
        norm = plt.Normalize(means.min(), means.max())
        colors = [cm.viridis(norm(val)) for val in means]
        cached_data = {
            'df': df,
            'today': today,
            'current_month': current_month,
            'current_YearMonth': current_YearMonth,
            'figsize': figsize,
            'colors': colors,
            'last_month_days_difference': last_month_days_difference,
            'total_months': total_months
        }
                   
        return cached_data


def main():
    # Get cached database instance
    db = get_database()
    
    # Load data if needed
    initialize_db(db)
    cached_data = mid_calc(db)
    st.title("Expense Dashboard")

    c11, c12 = st.columns(2)
    with c11:
        st.subheader("monthly_pie")
        st.pyplot(plots.monthly_pie(db,cached_data),use_container_width =False)
    with c12:
        st.subheader("new_monthly_pie")
        st.pyplot(plots.new_monthly_pie(db,cached_data),use_container_width =False)

    st.subheader("ytd_pivot_plot")
    st.pyplot(plots.ytd_pivot_plot(db,cached_data))

    c21, c22 = st.columns(2)
    with c21:
        st.subheader("util_plot")
        st.pyplot(plots.util_plot(db,cached_data))

    with c22:
        st.subheader("super_stacked")
        st.pyplot(plots.super_stacked(db,cached_data))
    
    st.subheader("ytd_pivot_table")
    st.dataframe(plots.ytd_pivot_table(db,cached_data).style.format("{:.0f}"))

    st.subheader("monthly_pivot_plot")
    fig = plots.monthly_pivot_multyplot(db,cached_data)
    st.pyplot(fig)

    st.subheader("Utilities Breakdown")
    
    # Supermarket Analysis
    st.subheader("Supermarket Spending")

    st.dataframe(plots.super_table(db,cached_data).style.format("{:.0f}"))

    st.subheader("get_super_targets")
    fig = plots.super_stacked(db,cached_data)
    st.pyplot(fig)

# figsize = (1,1)

if __name__ == "__main__":
    logger = setup_logging()
    main()
    logger.info("Done main")
