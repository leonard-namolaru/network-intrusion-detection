# pip install plotly
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import optuna
import optuna.visualization
import matplotlib.pyplot as plt
import seaborn as sns

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
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Cesser d'afficher chaque resultat d'essai
    
    etude = optuna.create_study(direction='maximize')

    # Executer 20 essais
    etude.optimize(k_plus_proches_voisins_score, n_trials=20)

    # Tracer les effets des hyperparams sur le score
    plot = optuna.visualization.plot_parallel_coordinate(etude)
    plot.show()
    # plot.savefig("KNN.png")

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
        max_colonnes = trial.suggest_int('features', 2, 51)

        classificateur = DecisionTreeClassifier(max_depth = max_profondeur, max_features = max_colonnes)
        classificateur.fit(x_entrainement, y_entrainement)

        return classificateur.score(x_validation, y_validation)


    # Executer l'etude
    etude = optuna.create_study(direction='maximize')
    etude.optimize(arbre_de_decision_score, n_trials=32)

    plot = optuna.visualization.plot_parallel_coordinate(etude)
    plot.show()
    #plot.savefig("arbres_decision.png")

    # Creer le modele avec les meilleurs parametres
    dt_modele = DecisionTreeClassifier(max_depth = etude.best_trial.params['features'], max_features = etude.best_trial.params['depth'])
    dt_modele.fit(x_entrainement, y_entrainement)

    # Score du modele
    y_pred = dt_modele.predict(x_validation)

    arbre_de_decision_score_entrainement = dt_modele.score(x_entrainement, y_entrainement)
    arbre_de_decision_score_validation = dt_modele.score(x_validation, y_validation)

    return (arbre_de_decision_score_entrainement, arbre_de_decision_score_validation, classification_report(y_validation, y_pred))

def forets_aleatoires(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple :
    rfc_modele = RandomForestClassifier(random_state = 0)
    params = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 10], "min_samples_leaf": [1, 2, 3]}

    # Utilisation de GridSearchCV pour trouver le meilleur ensemble d'hyperparametres
    gs = GridSearchCV(rfc_modele, params, cv=4, scoring='f1')
    gs.fit(pd.concat([x_entrainement, x_validation]), pd.concat([y_entrainement, y_validation]))

    df = pd.DataFrame(gs.cv_results_)
    sns.scatterplot(data=df, x='param_n_estimators', y='mean_test_score', style='param_max_depth', hue='param_min_samples_leaf')
    plt.show()
    plt.savefig("forets_aleatoires.png")

    rfc_modele = gs.best_estimator_

    # Score du modele
    y_pred = rfc_modele.predict(x_validation)

    random_forest_score_entrainement = rfc_modele.score(x_entrainement, y_entrainement)
    random_forest_score_validation = rfc_modele.score(x_validation, y_validation)

    return (random_forest_score_entrainement, random_forest_score_validation, classification_report(y_validation, y_pred))


def svm(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple :
    '''
    Support vector machine
    '''
    def svm_score(trial):
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf"])
        
        classificateur = SVC(kernel=kernel, gamma="scale", random_state=0)
        classificateur.fit(x_entrainement, y_entrainement)
        return classificateur.score(x_validation, y_validation)

    # Executer l'etude
    etude = optuna.create_study(direction='maximize')
    etude.optimize(svm_score, n_trials = 5)

    plot = optuna.visualization.plot_parallel_coordinate(etude).show()
    #plot.show()
    #plot.savefig("svm.png")

    # Creer le modele avec les meilleurs parametres
    modele = SVC(kernel = etude.best_trial.params['kernel'])
    modele.fit(x_entrainement, y_entrainement)

    # Score du modele
    y_pred = modele.predict(x_entrainement)

    svm_score_entrainement = modele.score(x_entrainement, y_entrainement)
    svm_score_validation = modele.score(x_validation, y_validation)

    return (svm_score_entrainement, svm_score_validation, classification_report(y_entrainement, y_pred))


def regression_logistique(x_entrainement : pd.DataFrame, y_entrainement : pd.DataFrame, x_validation : pd.DataFrame, y_validation : pd.DataFrame) -> tuple :
    params = {'C': [0.1, 1, 5, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'lbfgs']}

    modele_regression_logistique = LogisticRegression()

    # Creation d'un modele de regression logistique avec validation croisee
    grid = GridSearchCV(modele_regression_logistique, params, cv=4)

    # Entrainer et optimiser le modele
    grid.fit(pd.concat([x_entrainement, x_validation]), pd.concat([y_entrainement, y_validation]))

    df = pd.DataFrame(grid.cv_results_)
    sns.scatterplot(data=df, x='param_C', y='mean_test_score', style='param_penalty', hue='param_solver')
    #plt.show()
    plt.savefig("reg_log.png")

    modele_regression_logistique = grid.best_estimator_

    # Faire des predictions sur les donnees de validation puis evaluer les performances
    y_pred = modele_regression_logistique.predict(x_entrainement)

    regression_logistique_score_entrainement = modele_regression_logistique.score(x_entrainement, y_entrainement)
    regression_logistique_score_validation = modele_regression_logistique.score(x_validation, y_validation)

    return (regression_logistique_score_entrainement, regression_logistique_score_validation, classification_report(y_entrainement, y_pred))

if __name__ == '__main__' :
    x_entrainement, y_entrainement = creation_data_frame_division_x_y("entrainement.csv")
    x_validation, y_validation = creation_data_frame_division_x_y("validation.csv")
    x_test, y_test = creation_data_frame_division_x_y("test.csv")

    gnb_score_entrainement, gnb_score_validation, gnb_rapport = naive_bayes(x_entrainement, y_entrainement, x_validation, y_validation)
    print(f"**********NAIVE BAYES********** \n\n Score D'entrainement {gnb_score_entrainement} || Score de validation {gnb_score_validation} \n")
    print(gnb_rapport)

    voisins_score_entrainement, voisins_score_validation, voisins_rapport = k_plus_proches_voisins(x_entrainement, y_entrainement, x_validation, y_validation)
    print(f"**********K plus proches voisins********** \n\n Score D'entrainement {voisins_score_entrainement} || Score de validation {voisins_score_validation} \n")
    print(voisins_rapport)

    arbre_de_decision_score_entrainement, arbre_de_decision_score_validation, arbre_de_decision_rapport = arbre_de_decision(x_entrainement, y_entrainement, x_validation, y_validation)
    print(f"**********Arbre de decision********** \n\n Score D'entrainement {arbre_de_decision_score_entrainement} || Score de validation {arbre_de_decision_score_validation} \n")    
    print(arbre_de_decision_rapport)

    random_forest_score_entrainement, random_forest_score_validation, random_forest_rapport = forets_aleatoires(x_entrainement, y_entrainement, x_validation, y_validation)
    print(f"**********Forets aleatoires********** \n\n Score D'entrainement {random_forest_score_entrainement} || Score de validation {random_forest_score_validation} \n")    
    print(random_forest_rapport)

    svm_score_entrainement, svm_score_validation, svm_rapport = svm(x_entrainement, y_entrainement, x_validation, y_validation)
    print(f"**********SVM********** \n\n Score D'entrainement {svm_score_entrainement} || Score de validation {svm_score_validation} \n")    
    print(svm_rapport)

    regression_logistique_score_entrainement, regression_logistique_score_validation, regression_logistique_rapport = regression_logistique(x_entrainement, y_entrainement, x_validation, y_validation)
    print(f"**********Regression logistique********** \n\n Score D'entrainement {svm_score_entrainement} || Score de validation {svm_score_validation} \n")    
    print(regression_logistique_rapport)

    print("******************************* TEST ************************************")

    gnb_score_entrainement, gnb_score_test, gnb_rapport = naive_bayes(x_entrainement, y_entrainement, x_test, y_test)
    print(f"**********NAIVE BAYES********** \n\n Score D'entrainement {gnb_score_entrainement} || Score de test {gnb_score_test} \n")
    print(gnb_rapport)

    voisins_score_entrainement, voisins_score_test, voisins_rapport = k_plus_proches_voisins(x_entrainement, y_entrainement, x_test, y_test)
    print(f"**********K plus proches voisins********** \n\n Score D'entrainement {voisins_score_entrainement} || Score de test {voisins_score_test} \n")
    print(voisins_rapport)

    arbre_de_decision_score_entrainement, arbre_de_decision_score_test, arbre_de_decision_rapport = arbre_de_decision(x_entrainement, y_entrainement, x_test, y_test)
    print(f"**********Arbre de decision********** \n\n Score D'entrainement {arbre_de_decision_score_entrainement} || Score de test {arbre_de_decision_score_test} \n")    
    print(arbre_de_decision_rapport)

    random_forest_score_entrainement, random_forest_score_test, random_forest_rapport = forets_aleatoires(x_entrainement, y_entrainement, x_test, y_test)
    print(f"**********Forets aleatoires********** \n\n Score D'entrainement {random_forest_score_entrainement} || Score de test {random_forest_score_test} \n")    
    print(random_forest_rapport)

    svm_score_entrainement, svm_score_test, svm_rapport = svm(x_entrainement, y_entrainement, x_test, y_test)
    print(f"**********SVM********** \n\n Score D'entrainement {svm_score_entrainement} || Score de test {svm_score_test} \n")    
    print(svm_rapport)

    regression_logistique_score_entrainement, regression_logistique_score_test, regression_logistique_rapport = regression_logistique(x_entrainement, y_entrainement, x_test, y_test)
    print(f"**********Regression logistique********** \n\n Score D'entrainement {svm_score_entrainement} || Score de test {svm_score_test} \n")    
    print(regression_logistique_rapport)

