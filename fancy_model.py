# %%
import argparse
import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Train a ML model.')
parser.add_argument('--n_estimators', type=int, default=10,
                    help='Number of trees in the forest.')
parser.add_argument('--max_depth', type=int, default=2,
                    help='Number of lavels in a tree.')
args = parser.parse_args()

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
file_out = Path(f'data/mdl_{timestamp}.joblib')
file_out.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(mdl, file_out)

# %%
