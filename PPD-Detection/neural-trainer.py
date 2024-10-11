import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def trainer():
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(name='precision')])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=500, validation_split=0.2, callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    loss, accuracy, precision = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f'Test Accuracy: {accuracy*100:.2f}%')
    print(f'Test Precision: {precision*100:.2f}%')

    # Save the model
    model.save('ppd-detection-model.keras')

    # Save the scaler
    import joblib
    joblib.dump(scaler, 'scaler.joblib')

    print('Model and scaler saved successfully.')


# Function to make predictions with the neural network
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
    prediction_prob = model.predict(scaled_input)
    
    # Assuming a binary classification, threshold at 0.5
    prediction = (prediction_prob > 0.8).astype(int)
    
    # Convert prediction to 'Yes' or 'No' (or other class names as applicable)
    prediction_label = ['No' if p[0] == 0 else 'Yes' for p in prediction]

    return prediction_label[0], prediction_prob[0][0]

trainer()

# # Example new input
# new_input_example = {
#     'Age': '30-35',
#     'Irritable Towards Baby and Partner': 0,
#     'Problems Concentrating or Making Decision': 0,
#     'Feeling of Guilt': 0,
#     'Feeling sad or Tearful': 0,
#     'Trouble sleeping at night': 0,
#     'Overeating or loss of appetite': 0,
#     'Problems of bonding with baby': 0,
#     'Suicide attempt': 0
# }

# # Make a prediction using the neural network
# prediction_label, prediction_prob = predict_with_neural_network(new_input_example)
# print(f"The predicted feeling of anxiety is: '{prediction_label}' with a probability of {prediction_prob*100:.2f}%")