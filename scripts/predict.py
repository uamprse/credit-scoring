import pickle
import pandas as pd

def load_model_and_scaler(model_path, scaler_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict(filepath, model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=['number']).columns
    selected_params = [
        'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
        'Interest_Rate', 'Num_Credit_Inquiries',
        'Credit_Utilization_Ratio', 'Total_EMI_per_month'
    ]
    selected_params = [col for col in selected_params if col in numeric_cols]

    X = df[selected_params]

    predictions = model.predict(X)

    label_map = {
        0: 'Poor',
        1: 'Good',
        2: 'Standard'
    }
    predictions = [label_map.get(pred, pred) for pred in predictions]

    return predictions

