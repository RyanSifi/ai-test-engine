import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef)

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else '/app/_dataset/dataset_sucre.csv')
y = df['label_risk']
ID = ['file', 'class', 'method', 'profile']
# NB: aucune feature git retirée + split aléatoire => reproduit la fuite
X = df.drop(columns=ID + ['label_risk']).select_dtypes('number')

clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=0)
proba = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method='predict_proba')[:, 1]
pred = (proba >= 0.5).astype(int)

print('baseline (classe majoritaire) :', round(max(y.mean(), 1 - y.mean()), 3))
print('ROC-AUC  :', round(roc_auc_score(y, proba), 3))
print('PR-AUC   :', round(average_precision_score(y, proba), 3))
print('bal. acc :', round(balanced_accuracy_score(y, pred), 3))
print('MCC      :', round(matthews_corrcoef(y, pred), 3))