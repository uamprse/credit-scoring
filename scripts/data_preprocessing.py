import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    return pd.read_csv(filepath)


def clean_data(df):
    potential_numeric_columns = [
        'Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Outstanding_Debt', 'Credit_History_Age',
        'Amount_invested_monthly', 'Monthly_Balance'
    ]

    for col in potential_numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
