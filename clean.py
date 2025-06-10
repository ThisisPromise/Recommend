import pandas as pd
import re

def clean_dataset(df):
    df = df.drop_duplicates()
    
   
    df['Country'] = df['Country'].str.replace(r'XxY|鈽猴笍', '', regex=True)
    
    df['Quantity'] = df['Quantity'].astype(str).str.replace(r'[^0-9.-]', '', regex=True).astype(float)
    df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(r'[^0-9.-]', '', regex=True).astype(float)
    df['CustomerID'] = df['CustomerID'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
    df['CustomerID'] = pd.to_numeric(df['CustomerID'], errors='coerce')
    df['InvoiceNo'] = df['InvoiceNo'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
    df['StockCode'] = df['StockCode'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
    df['Description'] = df['Description'].str.replace(r'^\$', '', regex=True).str.strip()
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Description'] = df['Description'].fillna('Unknown')
    df['CustomerID'] = df['CustomerID'].fillna(0)
    
    return df


df = pd.read_csv('dataset.csv')


cleaned_df = clean_dataset(df)


cleaned_df.to_csv('cleaned_dataset.csv', index=False)