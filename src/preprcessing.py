import pandas as pd 
import numpy as np 


def data_loader(path):
    """
    خواندن دیتا 
    """
    df = pd.read_csv(path)
    df.drop('Price(USD)',axis=1 , inplace=True)
    return df

def df_dropna(df):
    """
    حذف داده های تکراری و داده های خالی
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    return df

def change_type(df):
    """
    عوض کردن تایپ ویژگی به خصوص متراژ
    """
    df['Area'] = df['Area'].astype(str).str.replace(',' ,'').str.strip().astype('int64')
    df['Elevator'] = df['Elevator'].astype('int8')
    df['Warehouse'] = df['Warehouse'].astype('int8')
    df['Room'] = df['Room'].astype('int8')
    df['Parking'] = df['Parking'].astype('int8')

    return df

def remove_outliers_and_log_price(df, column):
    """
    حذف اوتلایر ها 
    """


    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

def save_data(df , path):
    df.to_csv(path)


def main():
    """
    اجرا کردن پروژه 
    """
    path = '../data/raw/HomeTehran.csv'
    df = data_loader(path)
    df = df_dropna(df)
    df = change_type(df)
    df = remove_outliers_and_log_price(df , 'Area')
    df = remove_outliers_and_log_price(df , 'Price')
    path_save_data = '../data/preprocess/HomeTehran_Preprocess.csv'
    save_data(df , path_save_data)



if __name__ == '__main__':
    main()
