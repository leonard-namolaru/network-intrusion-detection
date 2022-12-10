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

def creation_data_frame_division_x_y(fichier_csv : str) -> tuple : 
    '''
    Creation d'un DataFrame et division en x et y.
    '''
    donnees = pd.read_csv(fichier_csv)
    x = donnees.drop(['class'], axis=1)
    y = donnees['class']

    return (x,y)

# standardize/scale data (convert it to z-scores), might be optional
#scaler = StandardScaler()
#x_train = scaler.fit_transform(x_train)
#validation = scaler.fit_transform(validation)
#test = scaler.fit_transform(test)

def naive_bayes(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple : 
    gnb_modele = GaussianNB()
    gnb_modele.fit(x_entrainement, y_entrainement)

    # Evaluation des performances
    y_pred = gnb_modele.predict(x_validation)

    gnb_score_entrainement = gnb_modele.score(x_entrainement, y_entrainement)
    gnb_score_validation = gnb_modele.score(x_validation, y_validation)

    return (gnb_score_entrainement, gnb_score_validation, classification_report(y_validation, y_pred))


def k_plus_proches_voisins(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple : 
    def k_plus_proches_voisins_score(trial):
        n = trial.suggest_int('knn', 2, 20)

        classificateur = KNeighborsClassifier(n)
        classificateur.fit(x_entrainement, y_entrainement)

        return classificateur.score(x_validation, y_validation)
    
    # Par defaut, Optuna affiche des messages de log
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Cesser d'afficher chaque résultat d'essai
    
    etude = optuna.create_study(direction='maximize')

    # Executer 16 essais
    etude.optimize(k_plus_proches_voisins_score, n_trials=20)

    # Creation du modele avec les parametres du meilleur essai
    knn_model = KNeighborsClassifier(n_neighbors=etude.best_trial.params['knn'])
    knn_model.fit(x_entrainement, y_entrainement)

    # Score
    y_pred = knn_model.predict(x_validation)
    voisins_score_entrainement = knn_model.score(x_entrainement, y_entrainement)
    voisins_score_validation = knn_model.score(x_validation, y_validation)

    return (voisins_score_entrainement, voisins_score_validation, classification_report(y_validation, y_pred))    


def arbre_de_decision(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple : 
    def arbre_de_decision_score(trial):
        max_profondeur = trial.suggest_int('depth', 2, 16)
        max_colonnes = trial.suggest_int('features', 2, 52)

        classificateur = DecisionTreeClassifier(max_depth = max_profondeur, max_features = max_colonnes)
        classificateur.fit(x_entrainement, y_entrainement)

        return classificateur.score(x_validation, y_validation)


    # Executer l'etude
    etude = optuna.create_study(direction='maximize')
    etude.optimize(arbre_de_decision_score, n_trials=16)
    optuna.visualization.plot_parallel_coordinate(etude).show()

    # Creer le modele avec les meilleurs parametres
    dt_model = DecisionTreeClassifier(max_depth = etude.best_trial.params['features'], max_features = etude.best_trial.params['depth'])
    dt_model.fit(x_entrainement, y_entrainement)

    # Score du modele
    y_pred = dt_model.predict(x_validation)

    arbre_de_decision_score_entrainement = dt_model.score(x_entrainement, y_entrainement)
    arbre_de_decision_score_validation = dt_model.score(x_validation, y_validation)

    fig = plt.figure(figsize= (50,30))
    tree.plot_tree(dt_model, filled=True)
    # plt.show()
    plt.savefig("arbre_de_decision")

    return (arbre_de_decision_score_entrainement, arbre_de_decision_score_validation, classification_report(y_validation, y_pred))


def random_forest(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> None :
    rfc_model = RandomForestClassifier(random_state = 0)
    params = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10]}

    # Utilisation de GridSearchCV pour trouver le meilleur ensemble d'hyperparametres
    gs = GridSearchCV(rfc_model, params, cv=4, scoring='f1')
    gs.fit(x_entrainement, y_entrainement)

    # Entrainement
    rfc_model.fit(x_entrainement, y_entrainement)

    y_pred = rfc_model.predict(x_validation)
    print( classification_report(y_validation, y_pred) )


def svm(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple :
    '''
    Support vector machine
    '''
    def svm_score(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        
        classificateur = SVC(kernel=kernel, gamma="scale", random_state=0)
        classificateur.fit(x_entrainement, y_entrainement)
        return classificateur.score(x_validation, y_validation)

    # run the study
    svm_study = optuna.create_study(direction='maximize')
    svm_study.optimize(svm_score, n_trials=8)
    # optuna.visualization.plot_optimization_history(svm_study).show()

    # make the model with best params
    svm_model = SVC(kernel=svm_study.best_trial.params['kernel'])
    svm_model.fit(x_entrainement, y_entrainement)

    # model score
    y_pred = svm_model.predict(x_entrainement)
    print(classification_report(y_entrainement, y_pred))

    svm_train_score = svm_model.score(x_entrainement, y_entrainement)
    svm_validate_score = svm_model.score(x_validation, y_validation)



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


if __name__ == '__main__' :
    x_entrainement, y_entrainement = creation_data_frame_division_x_y("entrainement_apres_selection_colonnes.csv")
    x_validation, y_validation = creation_data_frame_division_x_y("validation.csv")
    x_test, y_test = creation_data_frame_division_x_y("test.csv")

    gnb_score_entrainement, gnb_score_validation, gnb_rapport = naive_bayes(x_entrainement, y_entrainement, x_validation, y_validation)
    voisins_score_entrainement, voisins_score_validation, voisins_rapport = k_plus_proches_voisins(x_entrainement, y_entrainement, x_validation, y_validation)

    arbre_de_decision_score_entrainement, arbre_de_decision_score_validation, arbre_de_decision_rapport = (x_entrainement, y_entrainement, x_validation, y_validation)

