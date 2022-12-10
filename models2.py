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
import seaborn as sns

from warnings import simplefilter
simplefilter("ignore", category=Warning)

# import data and split into x and y
data = pd.read_csv("entrainement_apres_selection_colonnes.csv")
x_train = data.drop(['class'], axis=1)
y_train = data['class']

validation = pd.read_csv('validation.csv')
x_validate = validation.drop(['class'], axis=1)
y_validate = validation['class']

test = pd.read_csv('test.csv')
x_test = test.drop(['class'], axis=1)
y_test = test['class']


# standardize/scale data (convert it to z-scores), might be optional
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# validation = scaler.fit_transform(x_validate)
# test = scaler.fit_transform(x_test)


# Naive Bayes
def nb():
    print("******Naive Bayes******\n")
    gnb_model = GaussianNB()
    gnb_model.fit(x_train, y_train)

    # evaluate performance
    print(f"score entrainement: {gnb_model.score(x_train, y_train)}")
    y_pred = gnb_model.predict(x_test)
    print(classification_report(y_test, y_pred))

# silencing optuna trials
optuna.logging.set_verbosity(optuna.logging.WARNING)


# K-Nearest-Neighbours
# an objective function for optuna study
def KNN_score(trial):
    n = trial.suggest_int('knn', 2, 20)
    knn_classifier = KNeighborsClassifier(n)
    # personal comment: would be cool if the optuna study.best_trial would have the option to save the model so i dont have to retrain but oh well
    knn_classifier.fit(x_train, y_train)
    return knn_classifier.score(x_validate, y_validate)

def knn():
    print("******KNN******\n")

    knn_study = optuna.create_study(direction='maximize')
    # running 16 trials as i am unsure how to find an optimal number
    knn_study.optimize(KNN_score, n_trials=20)
    # plotting hyperparams effects on the score
    optuna.visualization.plot_parallel_coordinate(knn_study).show()

    # making the knn model with the parameters from the best trial
    knn_model = KNeighborsClassifier(
        n_neighbors=knn_study.best_trial.params['knn'])
    knn_model.fit(x_train, y_train)
    print(f"score entrainement: {knn_model.score(x_train, y_train)} \n score validation: {knn_model.score(x_validate, y_validate)}")
    # model score
    y_pred = knn_model.predict(x_test)
    print(classification_report(y_test, y_pred))


#   Decision tree (dt)

# again we make an objective function for optuna
def dt_score(trial):
    max_depth = trial.suggest_int('depth', 2, 16)
    max_features = trial.suggest_int('features', 2, 51)
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth, max_features=max_features)
    dt_classifier.fit(x_train, y_train)
    return dt_classifier.score(x_validate, y_validate)

def decision_trees():
    print("******Decision Trees******\n")
    # run the study
    dt_study = optuna.create_study(direction='maximize')
    dt_study.optimize(dt_score, n_trials=32)
    optuna.visualization.plot_parallel_coordinate(dt_study).show()

    # make the model with best params
    dt_model = DecisionTreeClassifier(
        max_depth=dt_study.best_trial.params['features'], max_features=dt_study.best_trial.params['depth'])
    dt_model.fit(x_train, y_train)
    # model score
    print(f"score entrainement: {dt_model.score(x_train, y_train)} \n score validation: {dt_model.score(x_validate, y_validate)}")
    y_pred = dt_model.predict(x_test)
    print(classification_report(y_test, y_pred))


# Random Forests
def random_forests():
    print("******Random Forests******\n")
    rfc_model = RandomForestClassifier(random_state=0)

    params = {'n_estimators': [100, 200, 500],
                'max_depth': [3, 5, 10],
                "min_samples_leaf": [1, 2, 3]}

    # Use GridSearchCV to find the best set of hyperparameters
    gs = GridSearchCV(rfc_model, params, cv=4, scoring='f1')
    gs.fit(pd.concat([x_train, x_validate]), pd.concat([y_train, y_validate]))
    
    df = pd.DataFrame(gs.cv_results_)
    sns.scatterplot(data=df, x='param_n_estimators', y='mean_test_score', style='param_max_depth', hue='param_min_samples_leaf')
    plt.show()

    rfc_model = gs.best_estimator_
    # model score
    print(f"score entrainement: {rfc_model.score(x_train, y_train)} \n score validation: {rfc_model.score(x_validate, y_validate)}")
    y_pred = rfc_model.predict(x_test)
    print(classification_report(y_test, y_pred))

# Support Vector Machines
# objective func for optuna study (the usual)
def svm_score(trial):
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
    svm_classifier = SVC(kernel=kernel, gamma="scale", random_state=0)
    svm_classifier.fit(x_train, y_train)
    return svm_classifier.score(x_validate, y_validate)

def svm():
    print("******SVM******\n")
    # run the study
    svm_study = optuna.create_study(direction='maximize')
    svm_study.optimize(svm_score, n_trials=5)
    optuna.visualization.plot_parallel_coordinate(svm_study).show()

    # make the model with best params
    svm_model = SVC(kernel=svm_study.best_trial.params['kernel'])
    svm_model.fit(x_train, y_train)
    # model score
    print(f"score entrainement: {svm_model.score(x_train, y_train)} \n score validation: {svm_model.score(x_validate, y_validate)}")
    y_pred = svm_model.predict(x_test)
    print(classification_report(y_test, y_pred))

# Logistic Regression (going for a different approach this time, not the same optuna study one)
def lg():
    print("******Logistic Regression******\n")

    params = {'C': [0.1, 1, 5, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'lbfgs']}

    lg_model = LogisticRegression()
    # create logistic regression model with cross-validation
    grid = GridSearchCV(lg_model, params, cv=4)

    # train and optimize the model
    grid.fit(pd.concat([x_train, x_validate]), pd.concat([y_train, y_validate]))

    # plot with scatterplot
    df = pd.DataFrame(grid.cv_results_)

    sns.scatterplot(data=df, x='param_C', y='mean_test_score', style='param_penalty', hue='param_solver')
    plt.show()

    lg_model = grid.best_estimator_
    # print the best params
    # print(model.best_params_)

    # make predictions on the testing data then evaluate performance
    print(f"score entrainement: {lg_model.score(x_train, y_train)} \n score validation: {lg_model.score(x_validate, y_validate)}")
    y_pred = lg_model.predict(x_test)
    print(classification_report(y_test, y_pred))


# call functions here 
nb()
knn()
svm()
random_forests()
decision_trees()
lg()