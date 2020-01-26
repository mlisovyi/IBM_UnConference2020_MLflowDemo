# %%
import joblib
import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# get the data
data = load_breast_cancer()
X, y = data['data'], data['target']
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=314)

# %%
# model parameters
n_estimators = 10
max_depth = 2

# train a model
mdl = RandomForestClassifier(n_estimators=n_estimators,  max_depth=max_depth)
mdl.fit(X_trn, y_trn)

# make predictions on hold-out dataset
y_pred = mdl.predict(X_tst)

# report model performance
accuracy = accuracy_score(y_tst, y_pred)
f1 = f1_score(y_tst, y_pred)
print(f'Model performance: accuracy = {accuracy:.3f}, F1 = {f1:.3f}')

# store the model
timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S')
joblib.dump(mdl, f'data/mdl_{timestamp}.joblib')

# %%
