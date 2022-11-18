from random import random, seed
from time import time

def division(jeu_de_donnees : str, graine : int, entrainement : str, validation : str, test : str) -> None :
    """
    Division des lignes du fichier de jeu de donnees en 3 fichiers : entrainement, validation et test, de maniere pseudo-aleatoire.
    Les fichiers de sortie sont approximativement equilibres (Les chances sont les mêmes pour chacune des lignes d'aller dans l'un des trois fichiers).
    
    Arguments:
        input: Chaines de caracteres, les noms de chacun des fichiers. Dans le cas du fichier de jeu de donnees, le nom doit etre le nom d'un fichier existant.
        graine: Un entier, la graine du generateur pseudo-aleatoire utilise.
    """
    fichier_jeu_de_donnees = open(jeu_de_donnees, "rb")

    fichier_entrainement = open(entrainement, "wb")
    fichier_validation = open(validation, "wb")
    fichier_test = open(test, "wb")

    seed(graine)

    i = 0
    for ligne in fichier_jeu_de_donnees :
        valeur_aleatoire = random() # random() -> x est dans l'intervalle [0, 1).

        if i != 0 :
            fichier = None
            if(valeur_aleatoire < (1 / 3)) :
                fichier = fichier_entrainement
            elif (valeur_aleatoire < (2 / 3)) :
                fichier = fichier_validation
            else :
                fichier = fichier_test

            fichier.write(ligne)

        else : # i == 0 => C'est la premiere ligne avec les noms des colonnes, donc cette ligne sera copiee dans les 3 fichiers
            fichier_entrainement.write(ligne)
            fichier_validation.write(ligne)
            fichier_test.write(ligne)

        i += 1

    fichier_jeu_de_donnees.close()
    fichier_entrainement.close()
    fichier_validation.close()
    fichier_test.close()

if __name__ == '__main__' :
    division('jeu_de_donnees.csv', time(), 'entrainement.csv', 'validation.csv', 'test.csv')