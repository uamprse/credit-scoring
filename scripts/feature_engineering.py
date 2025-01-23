import pandas as pd

def create_features(df):
    df['Income_to_Loan_Ratio'] = df['Annual_Income'] / (df['Num_of_Loan'] + 1)
    df['Debt_to_Income_Ratio'] = df['Outstanding_Debt'] / df['Annual_Income']
    df['Age_Bucket'] = pd.cut(df['Age'], bins=[18, 30, 50, 70], labels=['Young', 'Adult', 'Senior'])
    return df
