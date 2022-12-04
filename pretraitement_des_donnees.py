from pandas import DataFrame, read_csv, get_dummies
from sklearn.preprocessing import LabelEncoder, StandardScaler
from exploration_des_donnees import obtenir_noms_colonnes_csv, repartition_colonnes_selon_type_donnees_dans_colonne

def conversion_colonnes_nominales_en_colonnes_binaires(data_frame : DataFrame, noms_colonnes_nominales : list) -> DataFrame : 
    '''
    '''
    for nom_colonne in noms_colonnes_nominales :
        if  len( data_frame.groupby([nom_colonne]).size() ) == 2:
            data_frame[[nom_colonne]] = data_frame[[nom_colonne]].apply( LabelEncoder().fit_transform )
        else :
            data_frame = get_dummies(data_frame, columns=[nom_colonne])
    
    return data_frame

def suppression_colonnes_inutiles(data_frame : DataFrame, noms_colonnes : list) -> DataFrame : 
    '''
    '''
    for nom_colonne in noms_colonnes :
        if  len( data_frame.groupby([nom_colonne]).size() ) == 1:
                data_frame.drop(nom_colonne, axis = 1, inplace = True)
    
    return data_frame

def normalisation_min_max(data_frame : DataFrame, noms_colonnes : list) -> DataFrame : 
    '''
    '''
    for nom_colonne in noms_colonnes :
        max = data_frame[nom_colonne].max()
        min = data_frame[nom_colonne].min()
        data_frame[[nom_colonne]] = data_frame[[nom_colonne]].apply( lambda x : (x - min) / (max - min) )
    
    return data_frame

def normalisation(data_frame : DataFrame, noms_colonnes : list) -> DataFrame : 
    '''
    '''
    standard_scaler = StandardScaler().fit( data_frame[noms_colonnes] )   
    data_frame[noms_colonnes] = standard_scaler.transform( data_frame[noms_colonnes] ) 
    return data_frame


if __name__ == '__main__' :
    fichier_csv = 'jeu_de_donnees.csv'
    data_frame = read_csv( fichier_csv )

    noms_colonnes = obtenir_noms_colonnes_csv( fichier_csv )
    repartition_colonnes = repartition_colonnes_selon_type_donnees_dans_colonne(fichier_csv, noms_colonnes)

    data_frame = conversion_colonnes_nominales_en_colonnes_binaires(data_frame, repartition_colonnes['NOMINAL'])
    print( data_frame.head() )

    data_frame = suppression_colonnes_inutiles(data_frame, repartition_colonnes['BINAIRE'])
    print( data_frame.head() )
    print( data_frame.describe().transpose() )

    data_frame = normalisation(data_frame, repartition_colonnes['NUMERIQUE'])
    
    print( data_frame.head() )
    print( data_frame.describe().transpose() )

    data_frame.to_csv('jeu_de_donnees_apres_pretraitement.csv', index=False)


