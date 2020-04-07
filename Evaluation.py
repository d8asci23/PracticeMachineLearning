import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Import CSV file
titanic_df = pd.read_csv("/Users/jiwanhwang/Downloads/titanic/train.csv")


# Preprocessing
y_titanic_df = titanic_df['Survived']
X_titanic_df = titanic_df.drop('Survived', axis = 1)

## Fill NA
def fillna(df):
    df['Age'].fillna(df['Age'].mean(), inplace = True)
    df['Cabin'].fillna('N', inplace = True)
    df['Embarked'].fillna('N', inplace = True)
    df['Fare'].fillna(0, inplace = True)
    return df
## Remove unnecessary features
def drop_features(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df
## Label Encoding
def format_features(df):
    df['Cabin'] = df['Cabin'].str[:1]
    features = ['Cabin', 'Sex', 'Embarked']
    for feature in features:
        le=LabelEncoder()
        le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = format_features(df)
    return df


# Continue: Preprocessing using transform_features()
X_titanic_df = transform_features(X_titanic_df)


# X, y, test, train, Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_titanic_df, y_titanic_df, test_size = 0.2, random_state = 11)


# Define Evalute Function
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    f1 = f1_score(y_test, pred)
    print("Confusion Matrix")
    print(confusion)
    print("Accuracy: {}, Precision: {}, Recall: {}, f1: {}".format(accuracy, precision, recall, f1))

# Trade-off between precision and recall
from sklearn.preprocessing import Binarizer
### thresholds = [xx, xx, xx]
def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("Threshold: ", threshold)
        get_clf_eval(y_test, custom_predict)

#Prediction

## Make a classifier
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
## fit
lr_clf.fit(X_train, y_train)
## Prediction
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)
thresholds = [0.4, 0.45, 0.50, 0.55, 0.60]

# Get evaluate
print(get_clf_eval(y_test, pred))
print(get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1,1), thresholds))

# ROC curve
from sklearn.metrics import roc_curve
pred_proba_class1 = lr_clf.predict_proba(X_test)[:,1]
fprs, tprs, roc_thresholds = roc_curve(y_test, pred_proba_class1)
### Sampling
print(roc_thresholds)
thr_index = np.arange(0, roc_thresholds.shape[0], 5)
print("10 indexes for thresholds: ", thr_index)
print("10 sampled thresholds: ", np.round(roc_thresholds[thr_index], 2))
print("False Positive Rate(100-Specificity): ", np.round(fprs[thr_index],3))
print("True Positive Rate(Sensitivity): ", np.round(tprs[thr_index],3))

def roc_curve_plot(y_test, pred_proba_c1):
    fprs, tprs, thresholds = roc_curve(y_test, pred_proba_c1)
    # plot
    plt.plot(fprs, tprs, label='ROC')
    # 45 degree line
    plt.plot([0,1], [0,1], 'k--', label='Random')

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1),2))
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('FPR( 1 - Sensitivity )')
    plt.ylabel('TPR( Recall )')
    plt.legend()

roc_curve_plot(y_test, pred_proba_class1)
