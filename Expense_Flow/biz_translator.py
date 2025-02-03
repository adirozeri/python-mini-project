import requests
import pandas as pd
import warnings
from ast import literal_eval
warnings.filterwarnings('ignore')

def get_api_response_dict(business_name: str) -> dict:
    '''
    gets business name string
    returns ugly dict response
    '''
    print(business_name)
    url = 'https://places.googleapis.com/v1/places:searchText'
    api_key = 'AIzaSyAz5sOsxcD0zU2MNre2Kjnp8Om57WD6LMU'
    
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': api_key,
        'X-Goog-FieldMask': 'places.reviews,places.id,places.displayName,places.formattedAddress,places.types,places.primaryType'
    }
    
    data = {
        'textQuery': f'{business_name}'
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result
    # print(json.dumps(result, indent=2))
       
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None

def response_dict_to_nice_dict(biz_response_dict: dict) -> dict:
    '''
    gets: reponse dict
    returns: nice response dict
    '''
    biz_dict = {}
    
    # Check if we have the base places data
    if not biz_response_dict or 'places' not in biz_response_dict or not biz_response_dict['places']:
        biz_dict['displayName'] = 'MISSING'
        biz_dict['primaryType'] = 'MISSING'
        biz_dict['biz_reviews'] = []
        return biz_dict
    
    place = biz_response_dict['places'][0]  # Get first place
    
    # Handle displayName
    try:
        biz_dict['displayName'] = place['displayName']['text']
    except (KeyError, TypeError):
        biz_dict['displayName'] = 'MISSING'
    
    # Handle primaryType
    try:
        biz_dict['primaryType'] = place['primaryType']
    except KeyError:
        biz_dict['primaryType'] = 'MISSING'
    
    # Handle reviews
    biz_reviews = []
    try:
        for i, review in enumerate(place.get('reviews', [])):
            if i >= 5:  # Only take first 5 reviews
                break
            try:
                review_text = review['text']['text']
                biz_reviews.append(review_text)
            except (KeyError, TypeError):
                biz_reviews.append('MISSING_REVIEW')
    except (KeyError, TypeError):
        pass  # Keep empty list if no reviews
    
    biz_dict['biz_reviews'] = biz_reviews
    
    return biz_dict

def name_string_to_nice_dict_string(name: str) -> str:
    '''
    gets: business name string
    get info from api
    returns: nice dict string
    '''
    biz_response_dict = get_api_response_dict(name)
    biz_dict = response_dict_to_nice_dict(biz_response_dict)
    biz_dict['business_name'] = name
    return str(biz_dict)

def biz_df_to_biz_df_with_nice_dict_col(biz_names_df: pd.DataFrame) -> pd.DataFrame:
    '''
    get biz names df, add to df nice reponse string column for each biz
    '''
    unique_names_series = pd.DataFrame(biz_names_df['business_name'].dropna().unique())
    unique_names_series = unique_names_series.rename(columns={0:'business_name'})
    unique_names_series['response_text'] = unique_names_series['business_name'].apply(name_string_to_nice_dict_string)
    return unique_names_series

def melt_nice_dicts(api_res: pd.DataFrame) -> pd.DataFrame:
    '''
    gets: df with biz name column and nice dict response string coulmn
    returns: df with dict keys as columns
    '''
    #transform dictstring to dic object
    api_res['response_dict'] = api_res['response_text'].apply(literal_eval)
    api_res_df = pd.DataFrame(columns=['displayName','primaryType','biz_reviews','business_name'])
    
    # turning dicts to columns
    for sample_dict in api_res['response_dict']:
        if not sample_dict['biz_reviews']:
            sample_dict['biz_reviews'] = ['MISSING']
        api_res_df = pd.concat([api_res_df,pd.DataFrame(sample_dict)])

    # pivoting
    api_res_df['review_id'] = 'review_' + (api_res_df.groupby('displayName').cumcount()+1).astype(str)
    api_res_df_pivot = api_res_df.pivot(index=['business_name','displayName','primaryType'], columns='review_id', values='biz_reviews').reset_index()
    api_res_df_pivot = api_res_df_pivot[['business_name','displayName','primaryType','review_1','review_2','review_3','review_4','review_5']].reset_index(drop=True)
    api_res_df_pivot = api_res_df_pivot.fillna('MISSING')

    return api_res_df_pivot

def get_missing_businesses(df: pd.DataFrame, biz_info: pd.DataFrame) -> list:
    '''
    get two dfs, return missing biz list
    '''
    existing = biz_info['business_name'].unique()
    all_business = df['business_name'].unique()
    return [x for x in all_business if x not in existing]

def add_unseen_biz_api_info(df: pd.DataFrame) -> pd.DataFrame:
    '''
    gets: usualy dataframe from load_data() - this can have unseen businesses
    uses: 'all_business_df.csv' - all seen businesses
    returns: complete df
    exports: copmlete df to 'all_business_df.csv'
    '''
    # run: create existing df (from csv)
    business_df = pd.read_csv('all_business_df.csv')
    
    # run: create existing df (from csv) melt missing_biz
    missing_business_name = get_missing_businesses(df,business_df)

    if missing_business_name:
        missing_business_info = biz_df_to_biz_df_with_nice_dict_col(pd.DataFrame(missing_business_name, columns=['business_name']))
        missing_biz_info = melt_nice_dicts(missing_business_info)

        # concat
        business_df = pd.concat([business_df,missing_biz_info])

        # save to csv
    business_df.to_csv('all_business_df.csv', encoding='utf-8-sig', index=False)
    
    return business_df
