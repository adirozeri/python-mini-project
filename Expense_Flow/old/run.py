
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import logic
import db_engine
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="HomeBudget", page_icon="💰", layout="wide")



def initialize_database(logger):
    """Initialize database and load data if needed. This should only run once."""
    args = parse_args()
    db = db_engine.ExpenseDB()
    logger.info('initializing db - done')
    
    if args.loaddata:
        logger.info('getting data')
        df, biz_types_df = logic.load_data()
        logger.info('getting data - done')
        db.load_business_types(biz_types_df)
        db.load_expenses(df)
    
    return db



def setup_logging():
    """Configure rotating logging for the HomeBudget application"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure base logger
    logger = logging.getLogger('HomeBudget')
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



def render_dashboard(db):

    df = db.get_expenses()
    logic.mid_calc(df)

    c11, c12 = st.columns(2)
    with c11:
        st.subheader("monthly_pie")
        st.pyplot(logic.monthly_pie(df),use_container_width =False)
    with c12:
        st.subheader("new_monthly_pie")
        st.pyplot(logic.new_monthly_pie(df),use_container_width =False)

    st.subheader("ytd_pivot_plot")
    st.pyplot(logic.ytd_pivot_plot(df))

    c21, c22 = st.columns(2)
    with c21:
        st.subheader("util_plot")
        st.pyplot(logic.util_plot(df))

    with c22:
        st.subheader("super_stacked")
        st.pyplot(logic.super_stacked(df))
    
    st.subheader("ytd_pivot_table")
    st.dataframe(logic.ytd_pivot_table(df).style.format("{:.0f}"))

    st.subheader("monthly_pivot_plot")
    fig = logic.monthly_pivot_multyplot(df)
    st.pyplot(fig)

    st.subheader("Utilities Breakdown")
    
    # Supermarket Analysis
    st.subheader("Supermarket Spending")

    st.dataframe(logic.super_table(df).style.format("{:.0f}"))

    st.subheader("get_super_targets")
    fig = logic.super_stacked(df)
    st.pyplot(fig)



if __name__ == "__main__":
    logger = setup_logging()
    logger.info('Starting HomeBudget application')
    
    if 'db' not in st.session_state:
        logger.info('db is not in st.session_state')
        st.session_state.db = initialize_database(logger)
    
    render_dashboard(st.session_state.db)
