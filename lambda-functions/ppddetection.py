import json
import pandas as pd
import tensorflow as tf
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and preprocess the dataset
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

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Initialize models
dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.05, random_state=42)
rfc = RandomForestClassifier(random_state=42)
abc = AdaBoostClassifier(algorithm='SAMME', random_state=42)
cat = CatBoostClassifier(silent=True, random_state=42)
lgbm = LGBMClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
models = [dtc, rfc, abc, cat, lgbm, xgb]

# Train and evaluate models
model_performance = {}
for model in models:
    model_name = model.__class__.__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    y_score = model.predict_proba(X_test)[:, 1]  # Use the scores for the positive class
    roc_auc = roc_auc_score(y_test, y_score)
    model_performance[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC AUC': roc_auc}

def predict_feeling_anxious(new_input, model=rfc):
    input_df = pd.DataFrame([new_input])
    input_df.columns = input_df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")

    if 'age' in input_df.columns:
        input_df['age'] = input_df['age'].map(age_group)

    input_df = pd.get_dummies(input_df)

    missing_cols = set(X.columns) - set(input_df.columns)
    for c in missing_cols:
        input_df[c] = 0

    input_df = input_df[X.columns]
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    probability = prediction_proba[0][1] * 100

    # Decode the prediction to the original class label
    decoded_prediction = le.inverse_transform(prediction)[0]

    return (decoded_prediction, probability)

def predict_with_neural_network(new_input):
    model = tf.keras.models.load_model('ppd-detection-model.keras')
    scaler = joblib.load('scaler.joblib')
    # Convert new input to DataFrame
    input_df = pd.DataFrame([new_input])
    input_df.columns = input_df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")

    # Dummy coding for categorical variables (as done during training)
    input_df = pd.get_dummies(input_df)
    
    # Ensure the input has all the features the model was trained on, fill missing with 0
    trained_feature_names = [f for f in scaler.feature_names_in_]
    for col in trained_feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training data
    input_df = input_df[trained_feature_names]
    
    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)
    
    # Make a prediction
    prediction_prob = model.predict(scaled_input, verbose=0)
    
    # Assuming a binary classification, threshold at 0.5
    prediction = (prediction_prob > 0.5).astype(int)
    
    # Convert prediction to 'Yes' or 'No' (or other class names as applicable)
    prediction_label = ['No' if p[0] == 0 else 'Yes' for p in prediction]

    return prediction_label[0], prediction_prob[0][0]

def lambda_handler(event, context):
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

    models_prediction, models_prob = predict_feeling_anxious(input)
    neural_prediction, neural_prob = predict_with_neural_network(input)
    neural_prob = float(round(neural_prob*100, 2))

    print("Detections:")
    print(f"\tModels:")
    print(f"\t\tPrediction: {models_prediction} | Probability: {models_prob}%")
    print(f"\tNeural Network:")
    print(f"\t\tPrediction: {neural_prediction} | Probability: {neural_prob*100:.2f}%")

    if models_prediction == neural_prediction:
        if abs(float(models_prob) - float(neural_prob)) <= 10:
            return json({'message':'success', 'prediction':models_prediction})
        else:
            if (float(models_prob) - float(neural_prob)) >= 0:
                return json({'message':'success', 'prediction':models_prediction})
            else:
                return json({'message':'success', 'prediction':neural_prediction})
    else:
        if (float(models_prob) - float(neural_prob)) >= 0:
            return json({'message':'success', 'prediction':models_prediction})
        else:
            return json({'message':'success', 'prediction':neural_prediction})