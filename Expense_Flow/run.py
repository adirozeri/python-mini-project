import streamlit as st
st.set_page_config(
    page_title="HomeBudget",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import logic
# import db_engine
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import plots
import streamlit.components.v1 as components

hashed_password = stauth.Hasher(['659698']).generate()[0]

config = {
    'credentials': {
        'usernames': {
            'adir': {
                'name': 'John Doe',
                'password': hashed_password,
                'email': 'adirozeri@gmail.com'
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'some_signature_key_that_is_fairly_long_and_random',  # Make this more secure
        'name': 'homebudget_cookie'  # Give it a unique, app-specific name
    }
}

# Create the authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Add login UI




print("Script executing", datetime.now())
requirements = [
    'streamlit',
    'pandas',
    'matplotlib',
    'seaborn',
    'numpy',
    'python-dateutil'
]

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

@st.cache_data
def load_data():
    """Cache the data loading separately"""
    print('entered')
    return logic.load_data()

@st.cache_data
def mid_calc(df):
        # df = pd.read_sql("SELECT * FROM expenses", db.conn)
# 
        today = datetime.today()
        current_month = today.month
        current_YearMonth = today.strftime('%m-%Y')
        next_month = datetime.now() + relativedelta(months=1)
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
            'total_months': total_months,
            'next_month' : next_month
        }
                   
        return cached_data
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None



def edit_biz_types():
    st.title("Edit Business Types")

    # Load the CSV file
    biz_types = pd.read_csv('biz_types.csv', encoding='utf-8-sig')

    with st.form("edit_biz_types_form"):
        edited_df = st.data_editor(
            biz_types,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "business_name": st.column_config.TextColumn(
                    "Business Name"
                ),
                "business_type": st.column_config.TextColumn(
                    "Business Type"
                ),
                "business_utility": st.column_config.TextColumn(
                    "Business Utility"
                ),
                "super_name": st.column_config.TextColumn(
                    "Super Name"
                )
            }
        )

        submitted = st.form_submit_button("Save Changes")

        if submitted:
            edited_df.to_csv('biz_types.csv', index=False, encoding='utf-8-sig')
            st.success("Changes saved successfully!")
            st.cache_data.clear()
            # st.rerun()

    st.subheader("Current Business Types")
    st.dataframe(pd.read_csv('biz_types.csv', encoding='utf-8-sig'))

def inject_ga():
    ga_measurement_id = "G-2FRBVVZTEW"
    
    # Google Analytics tracking code
    ga_script = f"""
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id={ga_measurement_id}"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){{dataLayer.push(arguments);}}
        gtag('js', new Date());
        gtag('config', '{ga_measurement_id}');
    </script>
    """
    
    # Inject the script
    components.html(ga_script, height=0)





def main():
    if 'ga_initialized' not in st.session_state:
        inject_ga()
        st.session_state.ga_initialized = True

    name, authentication_status, username = authenticator.login('Login', 'main')
    if authentication_status:
        # Add logout button at the top
        authenticator.logout('Logout', 'main')

        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", ["Expense Dashboard", "Edit Business Types"])
        if page == "Expense Dashboard":

            df,_ = load_data()
            cached_data = mid_calc(df)
            st.title("Expense Dashboard")

            c11, c12 = st.columns(2)
            with c11:
                st.subheader("monthly_treemap")
                st.pyplot(plots.monthly_treemap(df),use_container_width =True)
            with c12:
                st.subheader("monthly_new_treemap")
                st.pyplot(plots.monthly_new_treemap(df),use_container_width =True)

            st.subheader("ytd_pivot_plot")
            # st.bar_chart(data=df.groupby(['charge_YearMonth','business_type'])['charge_nis'].sum().reset_index(), x='charge_YearMonth', y='charge_nis', color='business_type')
            st.pyplot(plots.ytd_pivot_plot(df,cached_data))

            st.subheader("daily_plot")
            st.pyplot(plots.daily_plot(df,cached_data))
            
            c21, c22 = st.columns(2)
            with c21:
                st.subheader("util_plot")
                st.pyplot(plots.util_plot(df,cached_data))

            with c22:
                st.subheader("super_stacked")
                st.pyplot(plots.super_stacked(df,cached_data))
            
            st.subheader("ytd_pivot_table")
            st.dataframe(plots.ytd_pivot_table(df,cached_data).style.format("{:.0f}"))

            st.subheader("monthly_pivot_plot")
            fig = plots.monthly_pivot_multyplot(df,cached_data)
            st.pyplot(fig)

            st.subheader("Utilities Breakdown")
            
            # Supermarket Analysis
            st.subheader("Supermarket Spending")

            st.dataframe(plots.super_table(df,cached_data).style.format("{:.0f}"))

            st.subheader("get_super_targets")
            fig = plots.super_stacked(df,cached_data)
            st.pyplot(fig)

        elif page == "Edit Business Types":
                edit_biz_types()

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    
    elif authentication_status == None:
        st.warning('Please enter your username and password')

# figsize = (1,1)

if __name__ == "__main__":
    print('***************************entered main***************************')
    logger = setup_logging()
    main()
    logger.info("Done main")
