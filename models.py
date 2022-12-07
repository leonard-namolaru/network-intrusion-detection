import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import tree
import optuna
import optuna.visualization
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter("ignore", category=Warning)

# import data and split into x and y
data = pd.read_csv("entrainement_apres_selection_colonnes.csv")
x_train = data.drop(['class'], axis=1)
y_train = data['class']

validation = pd.read_csv('test.csv')
x_validate = data.drop(['class'], axis=1)
y_validate = data['class']

test = pd.read_csv('validation.csv')


# standardize/scale data (convert it to z-scores), might be optional
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
validation = scaler.fit_transform(validation)
test = scaler.fit_transform(test)


# Naive Bayes
print("******Naive Bayes******\n")
gnb_model = GaussianNB()
gnb_model.fit(x_train, y_train)

# evaluate performance
y_pred = gnb_model.predict(x_validate)
print(classification_report(y_validate, y_pred))

# gnb_train_score = gnb_model.score(x_train, y_train)
# gnb_validate_score = gnb_model.score(x_validate, y_validate)
# print(
#     f"**********NAIVE BAYES********** \n\n Train score {gnb_train_score} || Validation score {gnb_validate_score} \n")

# K-Nearest-Neighbours
print("******KNN******\n")

# silencing optuna trials
optuna.logging.set_verbosity(optuna.logging.WARNING)

# an objective function for optuna study
def KNN_score(trial):
    n = trial.suggest_int('knn', 2, 20)
    knn_classifier = KNeighborsClassifier(n)
    # personal comment: would be cool if the optuna study.best_trial would have the option to save the model so i dont have to retrain but oh well
    knn_classifier.fit(x_train, y_train)
    return knn_classifier.score(x_validate, y_validate)


knn_study = optuna.create_study(direction='maximize')
# running 16 trials as i am unsure how to find an optimal number
knn_study.optimize(KNN_score, n_trials=20)

# making the knn model with the parameters from the best trial
knn_model = KNeighborsClassifier(
    n_neighbors=knn_study.best_trial.params['knn'])
knn_model.fit(x_train, y_train)
# model score
y_pred = knn_model.predict(x_validate)
print(classification_report(y_validate, y_pred))

# knn_train_score = knn_model.score(x_train, y_train)
# knn_validate_score = knn_model.score(x_validate, y_validate)
# print(
#     f"**********KNN********** \n\n Train score {knn_train_score} || Validation score {knn_validate_score} \n")


#   Decision tree (dt)
print("******Decision Trees******\n")
# again we make an objective function for optuna
def dt_score(trial):
    max_depth = trial.suggest_int('depth', 2, 16)
    max_features = trial.suggest_int('features', 2, 52)
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth, max_features=max_features)
    dt_classifier.fit(x_train, y_train)
    return dt_classifier.score(x_validate, y_validate)


# run the study
dt_study = optuna.create_study(direction='maximize')
dt_study.optimize(dt_score, n_trials=16)
optuna.visualization.plot_parallel_coordinate(dt_study).show()

# make the model with best params
dt_model = DecisionTreeClassifier(
    max_depth=dt_study.best_trial.params['features'], max_features=dt_study.best_trial.params['depth'])
dt_model.fit(x_train, y_train)
# model score
y_pred = dt_model.predict(x_validate)
print(classification_report(y_validate, y_pred))

# dt_train_score = dt_model.score(x_train, y_train)
# dt_validate_score = dt_model.score(x_validate, y_validate)
# print(
#    f"**********Decision Tree********** \n\n Train score {dt_train_score} || Validation score {dt_validate_score} \n")

# optional, plot the tree
# fig = plt.figure(figsize= (50,30))
# tree.plot_tree(dt_model, filled=True)
# plt.show()

# Random Forests
print("******Random Forests******\n")
rfc_model = RandomForestClassifier(random_state=0)

params = {'n_estimators': [100, 200, 500],
              'max_depth': [3, 5, 10]}

# Use GridSearchCV to find the best set of hyperparameters
gs = GridSearchCV(rfc_model, params, cv=4, scoring='f1')
gs.fit(x_train, y_train)

#train
rfc_model.fit(x_train, y_train)

# pred and report
y_pred = rfc_model.predict(x_validate)
print(classification_report(y_validate, y_pred))



# Support Vector Machines
print("******SVM******\n")
# objective func for optuna study (the usual)
def svm_score(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
    svm_classifier = SVC(kernel=kernel, gamma="scale", random_state=0)
    svm_classifier.fit(x_train, y_train)
    return svm_classifier.score(x_validate, y_validate)

# run the study
svm_study = optuna.create_study(direction='maximize')
svm_study.optimize(svm_score, n_trials=8)
# optuna.visualization.plot_optimization_history(svm_study).show()

# make the model with best params
svm_model = SVC(kernel=svm_study.best_trial.params['kernel'])
svm_model.fit(x_train, y_train)
# model score
y_pred = svm_model.predict(x_train)
print(classification_report(y_train, y_pred))

# svm_train_score = svm_model.score(x_train, y_train)
# # svm_validate_score = svm_model.score(x_validate, y_validate)
# print(
    # f"**********SVM********** \n\n Train score {svm_train_score} || Validation score {svm_validate_score} \n")

# Logistic Regression (going for a different approach this time, not the same optuna study one)
print("******Logistic Regression******\n")

params = {'C': [0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'lbfgs']}

lg_model = LogisticRegression()
# create logistic regression model with cross-validation
grid = GridSearchCV(lg_model, params, cv=4)

# train the model on the training data
grid.fit(x_train, y_train)

lg_model = grid.best_estimator_
# print the best params
# print(model.best_params_)

# make predictions on the testing data then evaluate performance
y_pred = lg_model.predict(x_validate)
# print(classification_report(y_validate, y_pred))