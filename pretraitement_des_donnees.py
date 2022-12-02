from pandas import read_csv, get_dummies
from sklearn.preprocessing import LabelEncoder

def conversion_colonnes_nominales_en_colonnes_numeriques(jeu_de_donnees : str, noms_colonnes_nominales : list) -> None : 
    '''
    '''
    data_frame = read_csv(jeu_de_donnees)

    for nom_colonne in noms_colonnes_nominales :
        if  len( data_frame.groupby([nom_colonne]).size() ) == 2:
            data_frame[[nom_colonne]] = data_frame[[nom_colonne]].apply( LabelEncoder().fit_transform )
        else :
            data_frame = get_dummies(data_frame, columns=[nom_colonne])
    
    return data_frame


if __name__ == '__main__' :
    print( conversion_colonnes_nominales_en_colonnes_numeriques('jeu_de_donnees.csv', ['protocol_type', 'service', 'flag', 'class']).sample(5) )
