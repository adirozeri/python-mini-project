# -*- coding: utf-8 -*-
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

figsize=(8,3)

# translator = Translator()

# %% [markdown]
# # data loading

# %% [markdown]
# ## proc file

# %%

#check
def proc_file(file_name):
    # '''log
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
    
    #valudation
#     print(biz_types.loc[:,biz_types.count(axis=0)>1])
#     print(biz_types[biz_types.duplicated(keep=False)])
    # if len(df[~df['business_name'].isin(biz_types['business_name'])])>0:
    #     logger.info('THERE ARE MISSING BIZ TYPES')
        # print(df[~df['business_name'].isin(biz_types['business_name'])])
#     return df[~df['business_name'].isin(biz_types['business_name'])]['business_name']
    
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

        # df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        df['charge' + '_YearMonth'] = df['charge_date'].dt.strftime('%m-%Y')#.astype('int32')
        df['purchase' + '_YearMonth'] = df['purchase_date'].dt.strftime('%m-%Y')#.astype('int32')
    df['days_in_month']=calendar.monthrange(df['charge_Year'].max(),df['charge_Month'].max())[1]
        # df.drop(fldname, axis=1, inplace=True)



# %% [markdown]
#
# ## proc data

# %%

def proc_data(df,biz_types):
    df = pd.merge(df, biz_types, how='left', left_on='business_name', right_on='business_name') #merge

    add_datepart(df) 
    df = df[df['charge_YearMonth']!='05-2024'].copy()
    df['plt_business_name'] = df['business_name'].transform(lambda x: x[::-1])
    df['charge_nis_day_average'] = df['charge_nis']/df['days_in_month']
    df = df.sort_values(by='charge_YearMonth', ascending=True)
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
    df = df.sort_values('charge_YearMonth', key=lambda x: pd.to_datetime(x.str.replace('-', ' '), format='%m %Y'))
    # print('load_data df.drop',df['charge_nis'].sum())
    # mid_calc(df)
    biz_info = biz_translator.get_biz_info_(df)[['business_name','primaryType']]
    
    df = pd.merge(df, biz_info, how='left', left_on='business_name', right_on='business_name') #merge
    return df, biz_types
# load_data()
# d.columns,d.groupby('charge_YearMonth')['charge_nis'].agg(['count','sum'])

# %%
# df,_ = load_data()


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

# %%


# %%
# today  = datetime(2024, 12, 5)

# # Subtract one month
last_month = relativedelta(months=1)
# print(relativedelta(months=1))  # Output: 2024-11-05

# %% [markdown]
# # calculation

# %% [markdown]
# ## total_mom_plot

# %%

def total_mom_plot(df):
    fig, ax = plt.subplots(figsize=figsize)  
    
    sums=df.groupby('charge_YearMonth')['charge_nis'].sum()
    norm = plt.Normalize(sums.min(), sums.max())
    colors = [cm.viridis(norm(val)) for val in sums]    
    sns.barplot(
        data=df,
        y='charge_nis',
        x='charge_YearMonth',
        estimator=np.sum,
        errorbar=None,
        palette = colors
    )
    ax.set_title('total_mom_plot')
    plt.xticks(rotation=90)
    ax.axhline(df['charge_nis'].sum()/total_months)
    ax.axhline(4000)
    plt.tight_layout()
    return fig

# _=total_mom_plot(df)

# %% [markdown]
# ## average_mom_plot

# %%
 
def average_mom_plot(df):
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)  
    
    sns.barplot(
        data=df,
        y='charge_nis_day_average',
        x='charge_YearMonth',
        estimator=np.sum,
        errorbar=None,
        palette = colors,
        # order='charge_YearMonth'
    )
    ax.set_title('average_mom_plot')
    ax.axhline(150)
    # plt.sort_xaxis()
    plt.xticks(rotation=90)

    plt.tight_layout()

    # Return the Axes in a list to match `get_targets`
    return fig
# df.columns
# a=average_mom_plot(df)

# %% [markdown]
# ## ytd_pivot_table

# %%
def ytd_pivot_table(df,biz_type_col = 'business_type'):
    data = df.groupby([biz_type_col,'charge_YearMonth'])['charge_nis'].agg(total_sum = 'sum').sort_values('total_sum',ascending=False)
    pivot_data = data.reset_index().pivot_table(values='total_sum', 
                                                index=biz_type_col, 
                                                columns='charge_YearMonth', 
                                                aggfunc='sum', fill_value=0)#.map(lambda x: f'{x:,.0f}')
    

    pivot_data.columns = pd.to_datetime(pivot_data.columns.map(lambda x: x.replace('-', ' ')), format='%m %Y')
    pivot_data = pivot_data.sort_index(axis=1, ascending=True)
    pivot_data.columns = pivot_data.columns.strftime('%m-%Y')  

    pivot_data['Total'] = pivot_data.sum(axis=1)
    pivot_data['Monthly Average'] = pivot_data['Total']/total_months
    pivot_data.loc['Total'] = pivot_data.sum(axis=0)

    return pivot_data
    
# ytd_pivot_table(df)

# %% [markdown]
# ## ytd_pivot_plot

# %%
def ytd_pivot_plot(df):

    fig,ax = plt.subplots(figsize=figsize)

    data = ytd_pivot_table(df)

    data = data.sort_values(by='Total',ascending=False).drop(columns = ['Monthly Average','Total'], index = ['Total'])
    # df = df.sort_index(axis=1)  
    # data = data / data.sum(axis=0) * 100

    bottom = np.zeros(len(data.columns))
    categories = data.index

    colors = cm.tab10_r(np.linspace(0, 1, len(categories)))
    # display(cm.viridis(categories))
    for i, (index, row) in enumerate(data.iterrows()):
        ax.bar(data.columns, row, 
                label=index,
                bottom=bottom,
                color=colors[i]
                )  # Use i as the index instead of undefined z
        bottom += row  # Update bottom for next stack
        # print(bottom)

    # print(bottom)
    # ax.set_ylim(0,105)
    ax.set_title('ytd_pivot_plot')
    fig.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Add labels and title
    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
# a=ytd_pivot_plot(df)

# %% [markdown]
# ## new_ytd_pivot_plot

# %%
def new_ytd_pivot_plot(df):

    fig,ax = plt.subplots()

    data = ytd_pivot_table(df,'primaryType')

    data = data.sort_values(by='Total',ascending=False).drop(columns = ['Monthly Average','Total'], index = ['Total'])
    # df = df.sort_index(axis=1)  
    # data = data / data.sum(axis=0) * 100
    # display(data)
    bottom = np.zeros(len(data.columns))
    categories = data.index
    # print(data.index)
    colors = cm.tab20(np.linspace(0, 1, len(categories)))
    # display(cm.viridis(categories))
    for i, (index, row) in enumerate(data.iterrows()):
        ax.bar(data.columns, row, 
                label=index,
                bottom=bottom,
                color=colors[i]
                )  # Use i as the index instead of undefined z
        bottom += row  # Update bottom for next stack
        # print(bottom)

    # print(bottom)
    # ax.set_ylim(0,105)
    ax.set_title('ytd_pivot_plot')
    fig.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Add labels and title
    # Show the plot
    plt.xticks(rotation=45)
    # plt.tight_layout()
    return fig
# _=new_ytd_pivot_plot(df)

# %% [markdown]
# ## monthly_pivot_multyplot

# %%
def monthly_pivot_multyplot(df):
    # plt.rcParams['figure.dpi'] = 100  
    avg_values = ytd_pivot_table(df)['Monthly Average']
    values_df  = ytd_pivot_table(df).drop(columns=['Total','Monthly Average'])
    tot_plots = len(values_df.index)
    colors = cm.viridis(np.linspace(0, 1, tot_plots))
    fig, axs = plt.subplots(1,tot_plots,figsize=(16, 5),sharey=True)

    for i, ax in enumerate(axs):
        curr_row = values_df.iloc[i]
        height = curr_row
        y=curr_row.index
        ax.barh(y,height,color=colors[i])
        # ax.set_title(height.name)
        ax.set_title(height.name)

        avg = avg_values.loc[height.name]
        ax.axvline(avg,label=avg)
        ax.tick_params(axis='y', length=0)
        

    plt.tight_layout()
    # plt.show()
    # ax = plt.gca()  
    return fig

# a=monthly_pivot_multyplot(df)

# %% [markdown]
# ## monthly_pie

# %%
# df.dtypes
# df['charge_date'].max().strftime('%m-%Y')

# %%
def monthly_pie(df):
    fig, _ = plt.subplots(figsize=figsize)
    current_YearMonth = df['charge_date'].max().strftime('%m-%Y')

    data = (df[df['charge_YearMonth']==current_YearMonth]
            .groupby(['business_type'])['charge_nis']
            .agg(total_sum = 'sum')
            .sort_values('total_sum', ascending=True)
       )

    colors = [cm.tab20(i/20) for i in range(len(data.total_sum))]
    
    plt.pie(data['total_sum'],
            labels=data.index,
            autopct='%1.1f%%',
            counterclock=True,
        #     pctdistance=0.75, 
            # labeldistance=0.5, 
            colors=colors,
            # rotatelabels=True
            )
    plt.title(f'monthly_table_pie for {current_YearMonth}')
    return fig
# a=monthly_pie(df)
# current_YearMonth

# %% [markdown]
# ## new_monthly_pie

# %%
def new_monthly_pie(df):
    fig, _ = plt.subplots(figsize=figsize)
    current_YearMonth = df['charge_date'].max().strftime('%m-%Y')
    data = (df[df['charge_YearMonth']==current_YearMonth]
            .groupby(['primaryType'])['charge_nis']
            .agg(total_sum = 'sum')
            .sort_values('total_sum', ascending=True)
       )

    
    colors = [cm.tab20(i/20) for i in range(len(data.total_sum))]

    
    plt.pie(data['total_sum'],
            labels=data.index,
            autopct='%1.1f%%',
            counterclock=True,
        #     pctdistance=0.75, 
            # labeldistance=0.5, 
            colors=colors,
            # rotatelabels=True
            )
    plt.title(f'monthly_table_pie for {current_YearMonth}')
    return fig
# a=new_monthly_pie(df)
# current_YearMonth

# %% [markdown]
# ## util_table

# %%
def util_table(df):
    data = df.groupby(['business_utility','charge_YearMonth'])['charge_nis'].agg(total_sum = 'sum').sort_values('total_sum',ascending=False)

    # sns.barplot(data=data, y='plt_business_name', x='total_sum', hue = 'charge_YearMonth', orient='h')
    pivot_data = data.reset_index().pivot_table(values='total_sum', 
                                                index='business_utility', 
                                                columns='charge_YearMonth', 
                                                aggfunc='sum', fill_value=0)
    pivot_data = pivot_data.drop(index = ['Not Utility'])
    
    pivot_data = pivot_data.sort_values(by=pivot_data.columns[-1],ascending = False)#.map(lambda x: f'{x:,.0f}')
    pivot_data.columns = pd.to_datetime(pivot_data.columns.map(lambda x: x.replace('-',' ')), format = '%m %Y')
    pivot_data = pivot_data.sort_index(axis=1, ascending = True)
    pivot_data.columns = pivot_data.columns.strftime('%m-%Y')
    return pivot_data

# util_table(df)

# %% [markdown]
# ## util_plot

# %%
def util_plot(df):


    fig,ax = plt.subplots(figsize=figsize)

    data = util_table(df)

    # data = data.sort_values(by='Total',ascending=False).drop(columns = ['Monthly Average','Total'], index = ['Total'])

    # data = data / data.sum(axis=0) * 100

    bottom = np.zeros(len(data.columns))
    categories = data.index
    colors = cm.tab10(np.linspace(0, 1, len(categories)))

    for i, (index, row) in enumerate(data.iterrows()):
        ax.bar(data.columns, row, 
                label=index,
                bottom=bottom,
                color=colors[i]
                )  # Use i as the index instead of undefined z
        bottom += row  # Update bottom for next stack
    fig.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # Add labels and title
    # Show the plot
    plt.title('util_plot')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
# a=util_plot(df)

# %% [markdown]
# ##  super_table

# %%
def super_table(df):
    data = df.groupby(['super_name','charge_YearMonth'])['charge_nis'].agg(total_sum = 'sum').sort_values('total_sum',ascending=False)

    # sns.barplot(data=data, y='plt_business_name', x='total_sum', hue = 'charge_YearMonth', orient='h')
    pivot_data = data.reset_index().pivot_table(values='total_sum', 
                                                index='super_name', 
                                                columns='charge_YearMonth', 
                                                aggfunc='sum', fill_value=0)#.map(lambda x: f'{x:,.0f}')
    # print(pivot_data.columns[-1])
    pivot_data = pivot_data.drop(index = ['Not Super'])

    pivot_data.columns = pd.to_datetime(pivot_data.columns.map(lambda x: x.replace('-',' ')))
    pivot_data = pivot_data.sort_index(axis=1, ascending = True)
    pivot_data.columns = pivot_data.columns.strftime('%m-%Y')

    pivot_data['Total'] = pivot_data.sum(axis=1)
    pivot_data['Monthly Average'] = pivot_data['Total']/total_months
    return pivot_data.sort_values(by=pivot_data.columns[-1],ascending = True)
# super_table(df)

# %% [markdown]
# ## super_stacked

# %%
def super_stacked(df):

    fig,ax = plt.subplots(figsize=figsize)

    data = super_table(df)

    data = data.sort_values(by='Total',ascending=False).drop(columns = ['Monthly Average','Total'])

    # data = data / data.sum(axis=0) * 100

    bottom = np.zeros(len(data.columns))
    categories = data.index

    colors = cm.Set1(np.linspace(0, 1, len(categories)))

    for i, (index, row) in enumerate(data.iterrows()):
        ax.bar(data.columns, row, 
                label=index,
                bottom=bottom,
                color=colors[i]
                )  # Use i as the index instead of undefined z
        bottom += row  # Update bottom for next stack
        # print(bottom)

    # print(bottom)
    # ax.set_ylim(0,105)
    fig.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.title('super_stacked')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig
# a=super_stacked(df)

# %% [markdown]
# ## super_split_plot

# %%

def super_split_multyplot(df):
    avg_values = super_table(df)['Monthly Average']
    values_df  = super_table(df).drop(columns=['Total','Monthly Average'])
    tot_plots = len(values_df.index)
    colors = cm.Dark2(np.linspace(0, 1, tot_plots))
    
    fig, axs = plt.subplots(1,tot_plots,figsize=(16, 5),sharey=True)

    for i, ax in enumerate(axs):
        curr_row = values_df.iloc[i]
        height = curr_row
        x=curr_row.index
        ax.barh(x,height,color=colors[i])
        ax.set_title(height.name)
        # ax.set_ylabel('NIS')

        avg = avg_values.loc[height.name]
        ax.axvline(avg,label=avg)
        # plt.xticks(rotation=45)
        

    plt.tight_layout()

    return fig

# a=super_split_multyplot(df)

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
    return df
    
def run_plots(df):
    total_mom_plot(df)
    average_mom_plot(df)
    ytd_pivot_table(df)
    ytd_pivot_plot(df)
    new_ytd_pivot_plot(df)
    # monthly_pivot_multyplot(df)
    monthly_pie(df)
    util_table(df)
    util_plot(df)
    super_table(df)
    super_stacked(df)
    # super_split_multyplot(df)

# %%
if __name__ == '__main__':
    df = main()
    print('done')
    run_plots(df)
    



