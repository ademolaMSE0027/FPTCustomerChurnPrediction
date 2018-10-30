import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score

pd.set_option('display.max_columns', None)

df = pd.read_csv('churn_data.csv')

# transform categorical features with binary variables to 0 and 1.
df.voice_mail_plan = df.voice_mail_plan.map({' no': 0, ' yes': 1})
df.intl_plan = df.intl_plan.map({' no': 0, ' yes': 1})
df.churned = df.churned.map({' False.': 0, ' True.': 1})

# Get the labels
y = df['churned'].values

# Drop the useless columns: 1. irrelevant featues 2. featuers with repeated info 3. the target variable
to_drop = ['state','account_length','area_code','phone_number','voice_mail_plan',
           'total_day_calls','total_day_minutes','total_eve_calls','total_eve_minutes',
           'total_night_calls','total_night_minutes','total_intl_calls',
           'total_intl_minutes','churned']

X = df.drop(to_drop, axis=1)

# Train/Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=1)


# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# fit the scaler to the training set and tranform both the training and test sets
X_train = pd.DataFrame(scaler.fit_transform(X_train.values), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test.values), columns=X.columns)

# ======================LogisticRegression===============================
lr = LogisticRegression()
C_grid = 0.001*10**(np.arange(0,1.01,0.01)*3)
parameters = {
    'penalty': ['l1', 'l2'],
    'C': C_grid
}

Grid_LR = GridSearchCV(lr, parameters, scoring='roc_auc')
Grid_LR.fit(X_train, y_train)

lr = Grid_LR.best_estimator_
print('Best score: ', Grid_LR.best_score_)
print('Best parameters set: \n', Grid_LR.best_params_)

y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:,1]

# ======================KNN====================================================

knn = KNeighborsClassifier()
parameters = {
    'n_neighbors':[4,8,16],
    'weights': ['uniform','distance']
}

Grid_KNN = GridSearchCV(knn, parameters, scoring='roc_auc')
Grid_KNN.fit(X_train, y_train)

knn = Grid_KNN.best_estimator_
print ('Best score: ', Grid_KNN.best_score_)
print ('Best parameters set: \n', Grid_KNN.best_params_)

y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)[:,1]

# ==========================Random Forest Classifier==============================
rf = RandomForestClassifier(n_estimators=20, criterion="entropy", random_state=0)
parameters = {
    "max_features": range(2,7),
    "min_samples_split": range(4,10),
    "min_samples_leaf": range(1,6),
}

Grid_RF = GridSearchCV(rf, parameters, scoring='roc_auc', n_jobs=-1)
Grid_RF.fit(X_train, y_train)

rf = Grid_RF.best_estimator_
print('Best score: ', Grid_RF.best_score_)
print('Best parameters set: \n', Grid_RF.best_params_)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:,1]

# =================================SVM============================================

svm = SVC(probability=True)
C_grid = 10**(np.arange(0,1.01,0.01)*2)
parameters = {'C': C_grid}

Grid_SVM = GridSearchCV(svm, parameters, scoring='roc_auc', n_jobs=-1)
Grid_SVM.fit(X_train, y_train)

svm = Grid_SVM.best_estimator_
print ('Best score: ', Grid_SVM.best_score_)
print ('Best parameters set: \n', Grid_SVM.best_params_)

y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:,1]


def cal_evaluation(classifier, cm, auc):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    f1 = 2 * precision * recall / (precision + recall)
    print (classifier)
    print ("Accuracy is " + str(accuracy))
    print ("Precision is " + str(precision))
    print ("Recall is " + str(recall))
    print ("F1 score is " + str(f1))
    # print ("ROC AUC is " + str(auc))


def draw_confusion_matrices(confusion_matricies):
    class_names = ['Not', 'Churn']
    for x in confusion_matricies:
        classifier, cm, auc = x[0], x[1], x[2]
        cal_evaluation(classifier, cm, auc)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, interpolation='nearest', cmap=plt.get_cmap('Reds'))


confusion_matrices = [
    ("Logisitic Regression", confusion_matrix(y_test, y_pred_lr), roc_auc_score(y_test, y_prob_lr)),
    ("K-Nearest-Neighbors", confusion_matrix(y_test, y_pred_knn), roc_auc_score(y_test, y_prob_knn)),
    ("Random Forest", confusion_matrix(y_test, y_pred_rf), roc_auc_score(y_test, y_prob_rf)),
    ("Support Vector Machine", confusion_matrix(y_test, y_pred_svm), roc_auc_score(y_test, y_prob_svm)),
]

draw_confusion_matrices(confusion_matrices)


