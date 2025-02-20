from pandas import DataFrame, read_csv
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from seaborn import heatmap
from matplotlib.pyplot import savefig, figure # python3 -m pip install -U matplotlib
from exploration_des_donnees import obtenir_noms_colonnes_csv, repartition_colonnes_selon_type_donnees_dans_colonne
from numpy import isnan

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

def correlation_classe_autres_colonnes(jeu_de_donnees : str) -> None : 
    '''
    '''
    figure(figsize=(20, 20))
    data_frame = read_csv(jeu_de_donnees)

    correlation = heatmap(data_frame.iloc[:,:].corr()[['class']].sort_values(by='class', ascending=False), linewidth=.5, cmap='Blues', annot=True, vmin=-1, vmax=1)
    correlation.set_title('Corrélation entre la classe et les autres colonnes', fontdict={'fontsize':12}, pad=12);
    savefig("correlation")

def discretisation(data_frame : DataFrame, noms_colonnes : list) -> DataFrame :
    '''
    '''
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    discretizer.fit( data_frame[noms_colonnes] )
    resultat = discretizer.transform( data_frame[noms_colonnes] )
    data_frame[noms_colonnes] = DataFrame(resultat)  

    return data_frame

def attribute_ratio(data_frame : DataFrame, colonnes_numeriques : list) -> DataFrame :
    classe = data_frame['class'] # DataFrame uniquement avec la colonne class 

    moyennes = data_frame[colonnes_numeriques].mean() # La moyenne de chacune des colonnes_numeriques

    # Pour chaque colonne, on verifie quelle est la moyenne de ses valeurs dans le cas ou la classe est 0
    # et quelle est la moyenne de ses valeurs dans le cas ou la classe est 1
    moyenne_par_classe = data_frame[ colonnes_numeriques + ['class']].groupby('class').mean()

    dictionnaire_attribute_ratio = dict()
    for colonne in colonnes_numeriques :
        # moyenne_par_classe[colonne] : la moyenne des valeurs de la colonne dans le cas ou la classe est 0
        #                               et la moyenne des valeurs de la colonne dans le cas ou la classe est 1
        # max(moyenne_par_classe[colonne]) : On prend le maximum entre la moyenne des valeurs de la colonne dans le cas ou la classe est 0 
        #                                    et entre la moyenne des valeurs de la colonne dans le cas ou la classe est 1
        # max(moyenne_par_classe[colonne]) / moyennes[colonne] : Nous divisons le maximum par la moyenne de toutes les valeurs de la colonne
        dictionnaire_attribute_ratio[colonne] = max(moyenne_par_classe[colonne]) / moyennes[colonne]
    
    colonnes_binaires = []
    for colonne in data_frame.columns :
        if colonne not in colonnes_numeriques and colonne != 'class' :
        # Toutes les colonnes qui ne sont pas numeriques sont binaires
            colonnes_binaires.append( colonne )

    dictionnaire = dict()
    for colonne in colonnes_binaires : # Pour chaque colonne binaire

        # data_frame[colonne] == 0 : Nous ne prenons que les lignes de la colonne ou la valeur de la colonne est 0
        # On compte, pour les lignes ou la valeur dans la colonne est 0, dans combien de cas la classe est 1, et dans combien de cas la classe est 0
        serie_0 = data_frame[ data_frame[colonne] == 0 ].groupby('class').size()

        # data_frame[colonne] == 1 : Nous ne prenons que les lignes de la colonne ou la valeur de la colonne est 1
        # On compte, pour les lignes ou la valeur dans la colonne est 1, dans combien de cas la classe est 1, et dans combien de cas la classe est 0
        serie_1 = data_frame[ data_frame[colonne] == 1 ].groupby('class').size()

        # serie_1 / serie_0 : Nous divisons le nombre de cas dans lesquels la valeur dans la colonne est 1 et la classe est 1, 
        #                     par le nombre de cas dans lesquels la valeur dans la colonne est 0 et la classe est 1
        #                     et nous divisons le nombre de cas dans lesquels la valeur dans la colonne est 1 et la classe est 0, 
        #                     par le nombre de cas dans lesquels la valeur dans la colonne est 0 et la classe est 0
        # max(serie_1 / serie_0) : Nous prenons le maximum entre les deux valeurs que nous avons calculees
        if max(serie_1 / serie_0) :
            dictionnaire_attribute_ratio[colonne] = max(serie_1 / serie_0)
        
        # Nous supprimons les valeurs NaN
        dictionnaire_attribute_ratio = dict((cle, valeur) for cle,valeur in dictionnaire_attribute_ratio.items() if not isnan(valeur))

        # Nous transformons le dictionnaire en une liste de tuples tries par les valeurs de l'attribut ratio
        dictionnaire = sorted(dictionnaire_attribute_ratio.items(), key = lambda x : x[1], reverse = True)

    colonnes = []
    for cle, valeur in dictionnaire :
        if valeur >= 0.01 :
            colonnes.append(cle)


    data_frame = data_frame[colonnes]
    # Nous voulons voir quelles colonnes numeriques il nous reste
    colonnes_numeriques_mis_a_jour = list(set(colonnes_numeriques).intersection(colonnes))

    # Nous normalisons a nouveau les colonnes numeriques
    standard_scaler = StandardScaler()
    data_frame[colonnes_numeriques_mis_a_jour] = standard_scaler.fit_transform(data_frame[colonnes_numeriques_mis_a_jour])

    data_frame['class'] = classe
    return data_frame

if __name__ == '__main__' :
    fichier_csv = 'jeu_de_donnees.csv'
    data_frame = read_csv( fichier_csv )

    noms_colonnes = obtenir_noms_colonnes_csv( fichier_csv )
    repartition_colonnes = repartition_colonnes_selon_type_donnees_dans_colonne(fichier_csv, noms_colonnes)

    fichier_csv = 'jeu_de_donnees_apres_pretraitement.csv'
    data_frame = read_csv( fichier_csv )

    print( data_frame.head() )
    data_frame = attribute_ratio(data_frame, repartition_colonnes['NUMERIQUE'])
    print( data_frame.head() )
    print( data_frame.columns )

    data_frame.to_csv('jeu_de_donnees_apres_pretraitement_selection_colonnes.csv', index=False)
    correlation_classe_autres_colonnes('jeu_de_donnees_apres_pretraitement_selection_colonnes.csv')