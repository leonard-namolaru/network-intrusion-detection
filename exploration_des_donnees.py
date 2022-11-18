def obtenir_noms_colonnes_csv(jeu_de_donnees : str) -> list :
    """
    La fonction recoit le nom d'un fichier csv et renvoie une liste avec les noms des colonnes.
    """
    fichier_jeu_de_donnees = open(jeu_de_donnees, "r")
    lignes = fichier_jeu_de_donnees.readlines()

    noms_colonnes = lignes[0].split(',')
    noms_colonnes[ len(noms_colonnes) - 1 ] = noms_colonnes[ len(noms_colonnes) - 1 ].split('\n')[0] # Suppression du caractere \n du nom de la derniere colonne

    fichier_jeu_de_donnees.close()
    return noms_colonnes

def repartition_noms_colonnes_selon_type_donnees_dans_colonne(jeu_de_donnees : str, noms_colonnes : list) -> dict :
    """
    La fonction recoit le nom d'un fichier csv et une liste avec les noms des colonnes (dans l'ordre dans lequel elles apparaissent dans le fichier). 
    La fonction renvoie un dictionnaire dans lequel les noms des colonnes sont divises selon le type de donnees dans chaque categorie :
    binaire (valeur numerique qui est 0 ou 1), numerique ou nominal.
    """

    fichier_jeu_de_donnees = open(jeu_de_donnees, "r")
    lignes = fichier_jeu_de_donnees.readlines()

    repartition_colonnes = {'BINAIRE' : [], 'NUMERIQUE' : [], 'NOMINAL' : []}

    donnees = []
    for i in range(1, len(lignes)) : # Nous commencons a partir de 1 car la ligne 0 contient les noms des colonnes
        donnees_ligne = lignes[i].split(',')  
        donnees_ligne[ len(donnees_ligne) - 1 ] = donnees_ligne[ len(donnees_ligne) - 1 ].split('\n')[0] # Suppression du caractere \n de la valeur de la derniere colonne
        donnees.append( donnees_ligne )

    # Nous parcourons la liste bidimensionnelle des donnees, colonne par colonne
    for j in range(0, len(noms_colonnes)) :
        binaire = True
        numerique = True
        nominal = True
        for i in range(0, len(donnees)) :

            if donnees[i][j].isnumeric() :
                nominal = False
                if int(donnees[i][j]) != 0 and int(donnees[i][j]) != 1 :
                    binaire = False
                else :
                  numerique = False  
            else :
                numerique = False
                binaire = False

            # Si nous avons pu exclure 2 options pour le type de donnees dans la colonne 
            # et puisque nous nous sommes retrouves avec le troisieme type, la recherche peut etre arretee pour cette colonne
            if int(nominal) + int(binaire) + int(numerique) == 1 :
                if nominal :
                    repartition_colonnes['NOMINAL'].append( noms_colonnes[j] )
                elif binaire :
                    repartition_colonnes['BINAIRE'].append( noms_colonnes[j] )
                else :
                    repartition_colonnes['NUMERIQUE'].append( noms_colonnes[j] )

                break # La recherche peut etre arretee pour cette colonne
    
    fichier_jeu_de_donnees.close()
    return repartition_colonnes

if __name__ == '__main__' :
    noms_colonnes = obtenir_noms_colonnes_csv('entrainement.csv')
    print( repartition_noms_colonnes_selon_type_donnees_dans_colonne('entrainement.csv', noms_colonnes) )