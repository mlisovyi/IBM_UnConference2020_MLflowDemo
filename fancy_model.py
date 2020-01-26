# %%
import argparse
import joblib
import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import neptune

parser = argparse.ArgumentParser(description='Train a ML model.')
parser.add_argument('--n_estimators', type=int, default=10,
                    help='Number of trees in the forest.')
parser.add_argument('--max_depth', type=int, default=2,
                    help='Number of lavels in a tree.')
parser.add_argument('--token_file', type=str, default='~/.ssh/neptune.creds',
                    help='The file containing neptune access token.')
args = parser.parse_args()


with open(agrs.token_file) as f:
    neptune_token, _ = f.read().split('\n')
neptune.init('mlisovyi/ibm_unconference2020_demo', api_token=neptune_token)

# get the data
data = load_breast_cancer()
X, y = data['data'], data['target']
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=314)

# %%
# train a model
mdl = RandomForestClassifier(n_estimators=args.n_estimators,  max_depth=args.max_depth, random_state=42)
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
