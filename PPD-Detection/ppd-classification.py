import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import StrMethodFormatter
from colorama import Fore, Style
from dython.nominal import associations

import lightgbm
import xgboost
import catboost
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV

from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.backend import clear_session
from tensorflow.config import list_physical_devices
from tensorflow.random import set_seed
set_seed(42)

print("Num GPUs Available: ", len(list_physical_devices('GPU')))

df = pd.read_csv('PPD-Detection\ppd-dataset.csv')
df.drop('Timestamp', axis=1, inplace=True)
df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("&", "and")

def summary_report_tree(model, X_test, y_test):
    """
    * Summary Report for Tree-based Models *
    """
    y_pred = model.predict(X_test)
    accuracy_ = accuracy_score(y_test, y_pred)
    precision_ = precision_score(y_test, y_pred)
    recall_ = recall_score(y_test, y_pred)
    f1_ = f1_score(y_test, y_pred)
    roc_auc_ = roc_auc_score(y_test, y_pred)
    return accuracy_, precision_, recall_, f1_, roc_auc_

def summary_report_nn(model, X_test, y_test):
    """
    * Summary Report for Neural Network-based Models *
    """
    y_pred = ((model.predict(X_test)>0.5)*1).flatten()
    accuracy_ = accuracy_score(y_test, y_pred)
    precision_ = precision_score(y_test, y_pred)
    recall_ = recall_score(y_test, y_pred)
    f1_ = f1_score(y_test, y_pred)
    roc_auc_ = roc_auc_score(y_test, y_pred)
    return accuracy_, precision_, recall_, f1_, roc_auc_

df = shuffle(df, random_state=42)
target = 'feeling_anxious'
X = df.drop(columns=target, axis=1).copy()
y = df[target].copy()
columns = ['irritable_towards_baby_and_partner',
    'problems_concentrating_or_making_decision',
    'feeling_of_guilt']
for name in columns:
    X[name] = X.groupby('age')[name].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))
age_group = {'25-30': 1,
    '30-35': 2,
    '35-40': 3,
    '40-45': 4,
    '45-50': 5}
X['age'] = X['age'].map(age_group)
X = pd.get_dummies(data = X, columns = X.columns[1:])
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=test_size, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train
dtc = DecisionTreeClassifier(max_depth=5, min_samples_leaf=0.05, random_state=42)
dtc.fit(X_train, y_train)
dtc.score(X_train, y_train)
dtc.score(X_test, y_test)
rfc = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_split=2 , min_samples_leaf=0.01 , random_state=42)
rfc.fit(X_train, y_train)
rfc.score(X_train, y_train)
rfc.score(X_test, y_test)
ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
ada.score(X_train, y_train)
ada.score(X_test, y_test)
lgb = lightgbm.LGBMClassifier(random_state=42)
lgb.fit(X_train, y_train)
lgb.score(X_train, y_train)
lgb.score(X_test, y_test)
xgb = xgboost.XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
xgb.score(X_train, y_train)
xgb.score(X_test, y_test)
cbc = catboost.CatBoostClassifier(verbose=0, random_state=42)
cbc.fit(X_train, y_train)
cbc.score(X_train, y_train)
cbc.score(X_test, y_test)
n_cols = X_train.shape[1]
model = Sequential()
model.add(Dense(40, input_shape = (n_cols,), activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()
X_train = X_train.astype('float32')
y_train = y_train.astype('float32')
model.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=10, verbose=0)
model.save('ppd-neural.keras')


'''
| Model          | Mean CV Score11 | Accuracy | Precision | Recall | F1-Score | ROC_AUC |
| -------------- | --------------- | -------- | --------- | ------ | -------- | ------- |
| CatBoost       | 0.979           | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |
| XGBoost        | 0.974           | 1.0      | 1.0       | 1.0    | 1.0      | 1.0     |
| LightGBM       | 0.976           | 0.993    | 1.0       | 0.990  | 0.995    | 0.995   |
| Neural Network | n.d.22          | 0.973    | 0.985     | 0.974  | 0.979    | 0.973   |
| Random Forest  | 0.900           | 0.870    | 0.894     | 0.908  | 0.901    | 0.854   |
| Decision Tree  | 0.878           | 0.877    | 0.904     | 0.908  | 0.906    | 0.864   |
| AdaBoost       | 0.849           | 0.844    | 0.853     | 0.918  | 0.885    | 0.812   |
'''

'''
| Model          | Set   | Accuracy | Precision | Recall | F1-score | ROC_AUC |
| -------------- | ----- | -------- | --------- | ------ | -------- | ------- |
| AdaBoost       | Train | 0.853    | 0.857     | 0.929  | 0.892    | 0.820   |
| AdaBoost       | Test  | 0.844    | 0.853     | 0.918  | 0.885    | 0.812   |
| CatBoost       | Train | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   |
| CatBoost       | Test  | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   |
| Decision Tree  | Train | 0.895    | 0.903     | 0.940  | 0.921    | 0.876   |
| Decision Tree  | Test  | 0.877    | 0.904     | 0.908  | 0.906    | 0.864   |
| LightGBM       | Train | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   |
| LightGBM       | Test  | 0.993    | 1.000     | 0.990  | 0.995    | 0.995   |
| Neural Network | Train | 0.994    | 0.996     | 0.995  | 0.996    | 0.994   |
| Neural Network | Test  | 0.973    | 0.985     | 0.974  | 0.979    | 0.973   |
| Random Forest  | Train | 0.911    | 0.923     | 0.943  | 0.932    | 0.897   |
| Random Forest  | Test  | 0.870    | 0.894     | 0.908  | 0.901    | 0.854   |
| XGBoost        | Train | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   |
| XGBoost        | Test  | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   |
'''