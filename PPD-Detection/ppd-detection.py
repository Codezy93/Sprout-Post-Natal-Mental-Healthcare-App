from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def preprocess_data():
    df = pd.read_csv('ppd-dataset.csv')
    df.drop('Timestamp', axis=1, inplace=True)
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")
    
    target = 'feeling_anxious'
    X = df.drop(columns=target, axis=1).copy()
    y = df[target].copy()

    columns = ['irritable_towards_baby_and_partner', 'problems_concentrating_or_making_decision', 'feeling_of_guilt',
            'feeling_sad_or_tearful', 'trouble_sleeping_at_night', 'overeating_or_loss_of_appetite',
            'problems_of_bonding_with_baby', 'suicide_attempt']

    for name in columns:
        X[name] = X.groupby('age')[name].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x.mean()))

    age_group = {'25-30': 1, '30-35': 2, '35-40': 3, '40-45': 4, '45-50': 5}
    X['age'] = X['age'].map(age_group)
    X = pd.get_dummies(data=X, columns=X.columns[X.dtypes == 'object'])

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    return X_train, X_test, y_train, y_test, le, X.columns

def preprocess_custom_input(custom_input, X_columns):
    input_df = pd.DataFrame([custom_input], columns=X_columns)
    age_group = {'25-30': 1, '30-35': 2, '35-40': 3, '40-45': 4, '45-50': 5}
    if 'age' in input_df.columns:
        input_df['age'] = input_df['age'].map(age_group)
    input_df = input_df.fillna(0)  # Replace NaNs with 0s for missing columns
    return input_df

def train_and_predict(model, X_train, y_train, X_test, y_test, custom_input_df):
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    print("Raw model output (probabilities):", y_score)  # Debugging line
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score)

    # Prediction on custom input
    custom_pred = model.predict(custom_input_df)[0]
    
    return {
        model_name: {
            'Prediction': custom_pred,
            'Accuracy': round(accuracy, 2),
            'Precision': round(precision, 2),
            'Recall': round(recall, 2),
            'F1 Score': round(f1, 2),
            'ROC AUC': round(roc_auc, 2)
        }
    }

def main(input_data):
    X_train, X_test, y_train, y_test, le, X_columns = preprocess_data()
    
    custom_input = {
        'age': input_data['age'],
        'irritable_towards_baby_and_partner': input_data['irr'],
        'problems_concentrating_or_making_decision': input_data['des'],
        'feeling_of_guilt': input_data['gui'],
        'feeling_sad_or_tearful': input_data['sad'],
        'trouble_sleeping_at_night': input_data['sle'],
        'overeating_or_loss_of_appetite': input_data['app'],
        'problems_of_bonding_with_baby': input_data['bon'],
        'suicide_attempt': input_data['sui']
    }
    
    custom_input_df = preprocess_custom_input(custom_input, X_columns)
    
    models = [
        DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.05, random_state=42),
        RandomForestClassifier(random_state=42),
        AdaBoostClassifier(algorithm='SAMME', random_state=42),
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        LGBMClassifier(random_state=42),
        CatBoostClassifier(silent=True, random_state=42)
    ]
    
    results = {}
    for model in models:
        result = train_and_predict(model, X_train, y_train, X_test, y_test, custom_input_df)
        results.update(result)
    
    # Decode the prediction to the original class label for all models
    for model_result in results.values():
        model_result['Prediction'] = le.inverse_transform([model_result['Prediction']])[0]

    
    #os.system('cls')

    return results


def predict_with_neural_network(new_input):
    model = tf.keras.models.load_model('ppd-detection-model.keras')
    scaler = joblib.load('scaler.joblib')
    input_df = pd.DataFrame([new_input])
    input_df.columns = input_df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")
    input_df = pd.get_dummies(input_df)

    trained_feature_names = [f for f in scaler.feature_names_in_]
    for col in trained_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[trained_feature_names]
    
    scaled_input = scaler.transform(input_df)
    prediction_prob = model.predict(scaled_input, verbose=0)
    print("Raw model output (probabilities):", prediction_prob)  # Debugging line

    prediction = (prediction_prob > 0.5).astype(int)
    prediction_label = ['No' if p[0] == 0 else 'Yes' for p in prediction]

    return prediction_label[0], prediction_prob[0][0]

def clear_screen():
    os.system('cls')

def lambda_handler(event):
    input = {
        'Age': str(event['age']),
        'Irritable Towards Baby and Partner': int(event['irr']),
        'Problems Concentrating or Making Decision': int(event['des']),
        'Feeling of Guilt': int(event['gui']),
        'Feeling sad or Tearful': int(event['sad']),
        'Trouble sleeping at night': int(event['sle']),
        'Overeating or loss of appetite': int(event['app']),
        'Problems of bonding with baby': int(event['bon']),
        'Suicide attempt': int(event['sui'])
    }

    neural_prediction, neural_prob = predict_with_neural_network(input)
    neural_prob = float(round(neural_prob*100, 2))
    results = main(input_data)
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")
    print(f"Neural Network: {neural_prediction} | {neural_prob}")

        
input_data = {
    "age": "30-35",  # A middle age group
    "irr": 0,  # Irritable Towards Baby and Partner: No
    "des": 0,  # Problems Concentrating or Making Decision: No
    "gui": 0,  # Feeling of Guilt: No
    "sad": 0,  # Feeling sad or Tearful: No
    "sle": 0,  # Trouble sleeping at night: No
    "app": 0,  # Overeating or loss of appetite: No
    "bon": 0,  # Problems of bonding with baby: No
    "sui": 0  # Suicide attempt: No
}

lambda_handler(input_data)