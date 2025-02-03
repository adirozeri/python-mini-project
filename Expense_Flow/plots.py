# %%
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd
# from googletrans import Translator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
import squarify


# %% [markdown]
# ## total_mom_plot

# %%

def total_mom_plot(db,cached_data):
    df = pd.read_sql("SELECT * FROM expenses", db.conn)

    fig, ax = plt.subplots(figsize=cached_data['figsize'])  
    
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
    ax.axhline(df['charge_nis'].sum()/cached_data['total_months'])
    ax.axhline(4000)
    plt.tight_layout()
    return fig

# _=total_mom_plot(df)

# %% [markdown]
# ## average_mom_plot

# %%
 
def average_mom_plot(db,cached_data):
    df = pd.read_sql("SELECT * FROM expenses", db.conn)
    # Create the plot
    fig, ax = plt.subplots(figsize=cached_data['figsize'])  
    
    sns.barplot(
        data=df,
        y='charge_nis_day_average',
        x='charge_YearMonth',
        estimator=np.sum,
        errorbar=None,
        palette = cached_data['colors'],
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
def ytd_pivot_table(df,cached_data,biz_type_col = 'business_type'):
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
    data = df.groupby([biz_type_col,'charge_YearMonth'])['charge_nis'].agg(total_sum = 'sum').sort_values('total_sum',ascending=False)
    pivot_data = data.reset_index().pivot_table(values='total_sum', 
                                                index=biz_type_col, 
                                                columns='charge_YearMonth', 
                                                aggfunc='sum', fill_value=0)#.map(lambda x: f'{x:,.0f}')
    

    pivot_data.columns = pd.to_datetime(pivot_data.columns.map(lambda x: x.replace('-', ' ')), format='%m %Y')
    pivot_data = pivot_data.sort_index(axis=1, ascending=True)
    pivot_data.columns = pivot_data.columns.strftime('%m-%Y')  

    pivot_data['Total'] = pivot_data.sum(axis=1)
    pivot_data['Monthly Average'] = pivot_data['Total']/cached_data['total_months']
    pivot_data.loc['Total'] = pivot_data.sum(axis=0)

    return pivot_data
    
# ytd_pivot_table(df)

# %% [markdown]
# ## ytd_pivot_plot

# %%
def ytd_pivot_plot(df,cached_data):
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
    fig,ax = plt.subplots(figsize=cached_data['figsize'])

    data = ytd_pivot_table(df,cached_data)

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
def new_ytd_pivot_plot(df,cached_data):

    fig,ax = plt.subplots()

    data = ytd_pivot_table(df,cached_data,'primaryType')

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
def monthly_pivot_multyplot(df,cached_data):
    # plt.rcParams['figure.dpi'] = 100  
    df=ytd_pivot_table(df,cached_data)
    avg_values = df['Monthly Average']
    values_df  = df.drop(columns=['Total','Monthly Average'])
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
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
    fig, _ = plt.subplots(figsize=(2,2))
    current_YearMonth = pd.to_datetime(df['charge_date']).max().strftime('%m-%Y')


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
# ## monthly_treemap


# %%
def monthly_treemap(df, figsize=(12, 8)):
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get current month's data
    current_YearMonth = df['charge_date'].max().strftime('%m-%Y')
    
    # Prepare data
    data = (df[df['charge_YearMonth'] == current_YearMonth]
            .groupby('primaryType')['charge_nis']
            .agg(total_sum='sum')
            .sort_values('total_sum', ascending=True)
            .reset_index())
    
    data['label'] = data['primaryType'] + '\n₪' + data.total_sum.apply(lambda x: f"{x:,.0f}")
    
    colors = sns.color_palette("tab20", len(data))
    
    plt.axis('off')
    
    squarify.plot(sizes=data['total_sum'],
                  label=data['label'],
                  color=colors,
                  pad=1,
                  ax=ax,
                  text_kwargs={'fontsize': 15})  # Adjust fontsize as needed
    
    plt.title(f'Monthly Expenses Distribution - {current_YearMonth}',
              pad=20,
              fontsize=14)
    
    return fig



# %%
def monthly_new_treemap(df, figsize=(12, 8)):
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get current month's data
    current_YearMonth = df['charge_date'].max().strftime('%m-%Y')
    
    # Prepare data
    data = (df[df['charge_YearMonth'] == current_YearMonth]
            .groupby('business_type')['charge_nis']
            .agg(total_sum='sum')
            .sort_values('total_sum', ascending=True)
            .reset_index())
    
    data['label'] = data['business_type'] + '\n₪' + data.total_sum.apply(lambda x: f"{x:,.0f}")
    
    colors = sns.color_palette("tab10", len(data))
    
    plt.axis('off')
    
    squarify.plot(sizes=data['total_sum'],
                  label=data['label'],
                  color=colors,
                  pad=1,
                  ax=ax,
                  text_kwargs={'fontsize': 15})  # Adjust fontsize as needed
    
    plt.title(f'Monthly Expenses Distribution - {current_YearMonth}',
              pad=20,
              fontsize=14)
        
    return fig



# %% [markdown]
# ## util_table

# %%
def util_table(df,cached_data):
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
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
def util_plot(df,cached_data):


    fig,ax = plt.subplots(figsize=cached_data['figsize'])

    data = util_table(df,cached_data)

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
def super_table(df,cached_data):
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
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
    pivot_data['Monthly Average'] = pivot_data['Total']/cached_data['total_months']
    return pivot_data.sort_values(by=pivot_data.columns[-1],ascending = True)
# super_table(db,cached_data)

# %% [markdown]
# ## super_stacked

# %%
def super_stacked(df,cached_data):

    fig,ax = plt.subplots(figsize=cached_data['figsize'])

    data = super_table(df,cached_data)

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

def super_split_multyplot(df,cached_data):
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
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

# a=super_split_multyplot(db,cached_data)

# %%
def daily_plot(df,cached_data):
    # df=db
    # df = pd.read_sql("SELECT * FROM expenses", db.conn)
    # purchase_date = pd.to_datetime(df['purchase_date']).dt.strftime('%d-%m-%Y')
    # current_YearMonth = df['charge_date'].max().strftime('%m-%Y')
    # pd.to_datetime(df['charge_YearMonth']).max
    data = df[
        (df['charge_YearMonth']==pd.to_datetime(df['charge_YearMonth']).max().strftime('%m-%Y')) & 
        (df['business_type']!='Bills')
    ][['business_type','purchase_date','charge_nis']].sort_values(by='purchase_date')
    data['formatted_date'] = pd.to_datetime(data['purchase_date']).dt.strftime('%d-%m')
    
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=cached_data['figsize'])

    # Plot the data
    sns.lineplot(data=data, 
                    x='formatted_date', 
                    y='charge_nis',
                    hue='business_type',
                    marker='o',
                    err_style=None)

    # plt.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    # ax.set_title(title)
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)

    plt.xticks(rotation=90)

    # ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.show()

    return fig
  
# if __name__ == '__main__': _ = daily_plot(df,cached_data)

# %% [markdown]
# #   main

# %%
def mid_calc(df):

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
    print('done mid_calc')   
    return cached_data

        
if __name__=='__main__':
    
    def main():
        # import logic
        # df,_ = logic.load_data()
        # df.to_csv('C:\\Users\\user\\Documents\\Python\\dffortesting.csv',encoding='utf-8-sig', index=False)
        df=pd.read_csv('C:\\Users\\user\\Documents\\Python\\dffortesting.csv',encoding='utf-8-sig')
        cached_data = mid_calc(df)
        current_YearMonth = pd.to_datetime(df['charge_date'].max()).strftime('%m-%Y')
        
        data = (df[df['charge_YearMonth'] == current_YearMonth]
                .groupby('primaryType')['charge_nis']
                .agg(total_sum='sum')
                .sort_values('total_sum', ascending=True)
                .reset_index())
        
        data['label'] = data['primaryType'] + '\n₪' + data.total_sum.apply(lambda x: f"{x:,.0f}")

    main()