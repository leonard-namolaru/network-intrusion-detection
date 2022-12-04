from pandas import DataFrame, read_csv
from sklearn.preprocessing import KBinsDiscretizer
from seaborn import heatmap
from matplotlib.pyplot import savefig, figure # python3 -m pip install -U matplotlib
from exploration_des_donnees import obtenir_noms_colonnes_csv, repartition_colonnes_selon_type_donnees_dans_colonne

def correlation_classe_autres_colonnes(jeu_de_donnees : str) -> None : 
    '''
    '''
    figure(figsize=(20, 20))
    data_frame = read_csv(jeu_de_donnees)

    correlation = heatmap(data_frame.iloc[:,:].corr()[['class']].sort_values(by='class', ascending=False), linewidth=.5, cmap='Blues', annot=True, vmin=-1, vmax=1)
    correlation.set_title('Corrélation entre la classe et les autres colonnes', fontdict={'fontsize':12}, pad=12);
    savefig("correlation")

def discretisation(data_frame : DataFrame, noms_colonnes : list) -> DataFrame :
    discretizer = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    discretizer.fit( data_frame[noms_colonnes] )
    resultat = discretizer.transform( data_frame[noms_colonnes] )
    data_frame[noms_colonnes] = DataFrame(resultat)  

    return data_frame


if __name__ == '__main__' :
    fichier_csv = 'jeu_de_donnees.csv'
    data_frame = read_csv( fichier_csv )

    noms_colonnes = obtenir_noms_colonnes_csv( fichier_csv )
    repartition_colonnes = repartition_colonnes_selon_type_donnees_dans_colonne(fichier_csv, noms_colonnes)

    fichier_csv = 'entrainement.csv'
    data_frame = read_csv( fichier_csv )

    data_frame = discretisation(data_frame, repartition_colonnes['NUMERIQUE'])
    print( data_frame.head() )

    correlation_classe_autres_colonnes('entrainement.csv')
