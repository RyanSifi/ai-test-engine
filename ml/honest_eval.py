import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             balanced_accuracy_score, matthews_corrcoef)

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else '/app/_dataset/dataset_sucre.csv')
y, groups = df['label_risk'], df['file']
LEAK = ['git_bugfix_count', 'git_total_commits', 'git_nb_authors', 'git_days_since_change']
ID   = ['file', 'class', 'method', 'profile']
X = df.drop(columns=ID + LEAK + ['label_risk']).select_dtypes('number')

clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=0)
proba = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups,
                          method='predict_proba')[:, 1]
pred = (proba >= 0.5).astype(int)

print('baseline (classe majoritaire) :', round(max(y.mean(), 1 - y.mean()), 3))
print('ROC-AUC  :', round(roc_auc_score(y, proba), 3))
print('PR-AUC   :', round(average_precision_score(y, proba), 3))
print('bal. acc :', round(balanced_accuracy_score(y, pred), 3))
print('MCC      :', round(matthews_corrcoef(y, pred), 3))