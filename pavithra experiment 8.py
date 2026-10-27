from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X,y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr,ytr)
ab = AdaBoostClassifier(n_estimators=50, random_state=42).fit(Xtr,ytr)

print('RF accuracy:', round(accuracy_score(yte, rf.predict(Xte)),4))
print('AdaBoost accuracy:', round(accuracy_score(yte, ab.predict(Xte)),4))
