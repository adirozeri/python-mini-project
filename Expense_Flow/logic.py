# %%
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# %%
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
# from googletrans import Translator
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import re
import calendar
import os
import pandas as pd
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import logging
import biz_translator

# %%
logger = logging.getLogger('HomeBudget')



def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        print(df)

# translator = Translator()

# %% [markdown]
# # data loading

# %% [markdown]
# ## proc file

# %%

#check
def proc_file(file_name):
    # '''
    # input - single csv file
    # output - main table

    # '''
    
    df_raw = pd.read_excel(file_name,header = None)
    df_raw['row_count'] = df_raw.count(axis=1)
    prev_was_0 = False
    table_id = -1
    for i,row in df_raw.iterrows():
        if row['row_count'] == 0:
            prev_was_0 = True
            df_raw.at[i, 'table_id'] = 999
        else:
            if prev_was_0:
                prev_was_0 = False
                table_id += 1
            df_raw.at[i, 'table_id'] = table_id
            
    df_raw = df_raw[df_raw['row_count'] != 999]
    df_raw['table_id']=df_raw['table_id'].rank(axis=0,method='dense',)

    israel_df, outbound_df = df_raw[df_raw['table_id']==4].copy(),df_raw[df_raw['table_id']==5].copy()
    # display(israel_df)
    # display(outbound_df)


    israel_df = proc_df(israel_df)
    # print('proc_file', israel_df['charge_nis'].sum())
    outbound_df = proc_df(outbound_df)
    # print('proc_file', outbound_df['charge_nis'].sum())

    df_israel_outbound = pd.concat([israel_df, outbound_df], ignore_index=True) #concat israel + outbound
    # print('proc_file', df_israel_outbound['charge_nis'].sum())
    return df_israel_outbound

# %%
# filenamee='csvs/excelNewBank (11).xlsx'
# a=proc_file(filenamee)

# %% [markdown]
# ## prod df
# 

# %%

#check
def proc_df(df):
    '''
    input - raw df of charges.
    this method will be called twice: 1. israel charges 2. outbound charges
    outout - same df  but with english col names and corrext dtypes
    '''
    
    df = df.reset_index(drop=True)
    df = df.rename(columns=df.iloc[2]).iloc[3:, :].reset_index(drop=True)
    
    cols = ['שם כרטיס', 'חיוב לתאריך', 'תאריך', 'שם בית עסק', "סכום חיוב בש''ח", "אסמכתא"]
    
    if cols[0] in df.columns:                 # if hasnt already been procceese
        df = df[cols]                         # reduces columns to only relevant
                                              # Create a dictionary for column renaming
        rename_cols = {
            'שם כרטיס': 'card_id',
            'חיוב לתאריך': 'charge_date',
            'תאריך': 'purchase_date',
            'שם בית עסק': 'business_name',
            "סכום חיוב בש''ח": 'charge_nis',
            'אסמכתא' : 'id'
        }

        # Rename columns using the dictionary
        df.rename(columns=rename_cols, inplace=True)
    
    # print(df.head())
    # print('asdasdad',df.columns)
    # print(df['charge_date'])
    df = df.dropna(subset=['charge_nis'])
    df['charge_date'] = pd.to_datetime(df['charge_date'])
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['charge_nis'] = df['charge_nis'].astype('float32')
    # df['purchase_nis'] = df['purchase_nis'].astype('float32')
    df['card_id'] = df['card_id'].astype('category')
    df['business_name'] = df['business_name'].astype('category')
    # display('proc_df',df['charge_nis'].sum())
    return df

# %%
# filenamee='csvs/excelNewBank (11).xlsx'
# a=proc_file(filenamee)
# a['charge_nis'].sum()

# %% [markdown]
# ## proc biz

# %%
#check
def proc_biz(df, biz_file = 'biz_types.csv'):
    '''
    input - biz csv
    output - 1. (biz name, biz type) df 
             2. dictionart - biz type = [biz names] 
    '''
    biz_types = pd.read_csv(biz_file)

    biz_types=biz_types.iloc[:,:4]
    biz_types = biz_types.drop_duplicates().reset_index(drop=True)
    dup_biz = biz_types[biz_types.duplicated(subset = 'business_name',keep=False)]
    if dup_biz.size > 2:
        logger.error('biz types not configure correctly - duplicates exist')
        logger.error(dup_biz)
        dup_biz.to_csv("duplivated_biz.csv", index=False)
        raise RuntimeError('duplicate business types')
        
    types_dict = {
            'חשבונות' : 'Bills',
            'סופר' : 'Supermarket',
            'בילוי' : 'Fun',
            'אחר' : 'Other',
            'פארם' : 'Pharma',
            'קיוסק' : 'Kiosk',
            'מוניות' : 'Taxies'

    }

    utils_types_dict = {
            'אינטרנט' : 'Internet',
            'ארנונה' : 'Arnona',
            'ביטוח' : 'Insurances',
            'אחר' : 'Not Utility',
            'גז' : 'Gas',
            'ועד' : 'Vaad',
            'חשמל' : 'Electricity',
            'טלויזיה' : 'TV',
            'מים' : 'Water',
            'ספוטיפיי' : 'Spotify',

    }

    super_name = {
                'אחר' : 'Not Super',
                'אמפמ' : 'Ampm',
                'טיבטעם' : 'TivTaam',
                'ירקות' : 'Vegtables',
                'מאפייה' : 'Bakery',
                'סופר אחר' : 'Other Super',
                'סופר פלוס' : 'Super Plus',
                'קצבדגים' : 'MeetorFish'

    }
   
    biz_types['business_type']=biz_types['business_type'].map(types_dict)
    biz_types['business_utility']=biz_types['business_utility'].map(utils_types_dict)
    biz_types['super_name']=biz_types['super_name'].map(super_name)
    
    
    return biz_types, df[~df['business_name'].isin(biz_types['business_name'])]['business_name']
# proc_biz()

# a= load_data()

# %%


# %% [markdown]
# ## add datepart
# 

# %%

def add_datepart(df):
    '''
    adds date parts to df
    '''
    attr = ['days_in_month','Year',  'Day', 'Dayofweek', 'Dayofyear','Month']
    date_cols = ['charge_date']#, 'purchase_date']
    for fldname in date_cols:
        fld = df[fldname]
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        for n in attr: 
            df[targ_pre + n] = getattr(fld.dt, n.lower())

    df['charge' + '_YearMonth'] = df['charge_date'].dt.strftime('%m-%Y')#.astype('int32')
    df['purchase' + '_YearMonth'] = df['purchase_date'].dt.strftime('%m-%Y')#.astype('int32')
    df['days_in_month']=calendar.monthrange(df['charge_Year'].max(),df['charge_Month'].max())[1]
    
    return df




# %% [markdown]
# 
# ## proc data

# %%

def proc_data(df,biz_types):
    df = pd.merge(df, biz_types, how='left', left_on='business_name', right_on='business_name') #merge

    df = add_datepart(df) 
    df = df[df['charge_YearMonth']!='05-2024'].copy()
    df['plt_business_name'] = df['business_name'].transform(lambda x: x[::-1])
    df['charge_nis_day_average'] = df['charge_nis']/df['days_in_month']
    df = df.sort_values(by='charge_YearMonth', ascending=True)
    df['charge_daydate'] = pd.to_datetime(df['charge_date']).dt.strftime('%d-%m')
    # print(df['charge_daydate'])
    return df
    


# %% [markdown]
# ## biz translator

# %%


# %%


# %% [markdown]
# ## load data

# %%

def load_data():
    missing = pd.DataFrame()
    folder = Path('csvs')
    df = pd.DataFrame()

    for file_name in folder.iterdir():
        if ((file_name.is_file()) and (file_name.suffix == '.xlsx')):
            logger.info(file_name.stem)
            # print('load_data',file_name.stem)
            df_israel_outbound = proc_file(file_name)                               
            df = pd.concat([df, df_israel_outbound], ignore_index=True)
    # print('load_data concat sum',df['charge_nis'].sum())        
    biz_types, missing = proc_biz(df, biz_file='biz_types.csv')
    # print('load_data proc_biz',df['charge_nis'].sum())        
    if not missing.empty:
            logger.error('There are missing biz types - aborting')
            save_missing(missing)
            raise RuntimeError('Missing business types')
    
    df = df.drop_duplicates()
    df = df.drop(columns=['id'])
    # print('load_data df.drop',df['charge_nis'].sum())
    df = proc_data(df,biz_types)
    # print(df.columns)
    df = df.sort_values('charge_YearMonth', key=lambda x: pd.to_datetime(x.str.replace('-', ' '), format='%m %Y'))
    # print('load_data df.drop',df['charge_nis'].sum())
    # mid_calc(df)
    biz_info = biz_translator.add_unseen_biz_api_info(df)[['business_name','primaryType']]
    
    df = pd.merge(df, biz_info, how='left', left_on='business_name', right_on='business_name') #merge
    return df, biz_types
# if __name__ == '__main__': df,_ =  load_data()
# d.columns,d.groupby('charge_YearMonth')['charge_nis'].agg(['count','sum'])

# %%
# if __name__ == '__main__': print(df.columns)

# %%
def save_missing(missing_df):
    # Create missing directory if it doesn't exist
    missing_dir = Path('missing')
    missing_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = missing_dir / f'missing_{timestamp}.csv'
    
    # Save file
    missing_df.to_csv(filename, encoding='utf-8-sig', index=False)
    logger.info(f"Saved missing businesses to {filename}")



# %%

def run_load_data():
    missing = pd.DataFrame()
    print("run_load_date")
    df,biz_types_df,missing = load_data()
    if not missing.empty:
        print('all missing values')
        print(missing)
        save_missing(missing)
        return df,biz_types_df,missing
    # df.to_csv('main_df.csv',encoding='utf-8-sig', index=False)

    
    return df,biz_types_df,missing
# print(run_load_data()[0].shape)



# %%

COMMANDS = {
    '/getytdpivot': 'tbd',
    '/get_totals': 'tbd',
    '/help': 'tbd',
    '/start': 'tbd'
}

# %% [markdown]
# # main

# %%
def mid_calc(df):
    global today
    global current_month
    global current_YearMonth
    global figsize
    global colors
    global last_month_days_difference
    global total_months
    
    today = datetime.today()
    current_month = today.month
    current_YearMonth = datetime.now().strftime('%m-%Y')
    last_month_total_days = (datetime.today() - relativedelta(months=1)).replace(day=15)
    last_month_days_difference = (today - last_month_total_days).days

    total_days_df = df[['charge_YearMonth','charge_days_in_month']].drop_duplicates()#['charge_days_in_month'].agg(['count','sum'])
    total_days_without_last_month = total_days_df[total_days_df['charge_YearMonth']!=current_YearMonth]['charge_days_in_month'].sum()
    total_months = (total_days_without_last_month + last_month_days_difference)/(365/12)
    figsize=(5,3)

    means=df.groupby('charge_YearMonth')['charge_nis_day_average'].sum()
    norm = plt.Normalize(means.min(), means.max())
    colors = [cm.viridis(norm(val)) for val in means]

# mid_calc(df)

# %%

def main():
    df, _ = load_data()
    mid_calc(df)
    # run_plots(df)
    # import plotly.express as px
    # import numpy as np
    # # df = px.data.gapminder().query("year == 2007")
    # fig = px.treemap(df, path=[px.Constant("business_type")], values='charge_nis',
    #                 color='charge_nis',#, hover_data=['iso_alpha'],
    #                 color_continuous_scale='RdBu',
    #                 # color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop'])
    #                 )
    # fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    # fig.show()
    return df
    


# %%
if __name__ == '__main__':
    df = main()
    # ytd_pivot_plot(df)
    print('done')

    # run_plots(df)
    