import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

X = np.array([[2,3], [1,2], [3,1], [6,5], [7,8],[8,6]])
y = np.array([0,0,0,1,1,1])
model = LogisticRegression().fit(X,y)
y_pred = model.predict(X)
probs = model.predict_proba(X)[:,1]

print('Accuracy:', accuracy_score(y,y_pred))
print('Precision:', precision_score(y,y_pred))
print('Recall:', recall_score(y,y_pred))
print('F1:', f1_score(y,y_pred))
print('AUC:', roc_auc_score(y,probs))

fpr, tpr, _ = roc_curve(y, probs)
plt.plot(fpr, tpr); plt.plot([0,1], [0,1],'--') 
plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('ROC Curve'); plt.show()
