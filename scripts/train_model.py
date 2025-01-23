import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scripts.data_preprocessing import load_data, clean_data
from scripts.evaluate import evaluate_model
from scripts.feature_engineering import create_features
from scripts.predict import *


def train_model(filepath, target_column):

    df = load_data(filepath)
    df = clean_data(df)
    df = create_features(df)

    numeric_cols = df.select_dtypes(include=['number']).columns
    selected_params = [
        'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_Credit_Inquiries',
        'Credit_Utilization_Ratio', 'Total_EMI_per_month'
    ]
    selected_params = [col for col in selected_params if col in numeric_cols]
    print("Выбранные гиперпараметры:", selected_params)

    X = df[selected_params]
    y = df[target_column].map({'Poor': 0, 'Standard': 1, 'Good': 2})
    print(df.head(5))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    if not os.path.exists('models'):
        os.makedirs('models')

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Модель обучена и сохранена.")


if __name__ == "__main__":
    train_model("/data/train.csv", "Credit_Score")

    predictions = predict("/data/test.csv", "models/model.pkl")
    print("Предсказания:", predictions)