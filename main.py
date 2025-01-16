from generation_features import caraxteristique_data
import pandas as pd
import os
os.environ['TCL_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
import concurrent.futures


"""main est utilisé principalement pour la génération des données utiles pour l'entrainement du modèle de machine learning"""

caracteristique = caraxteristique_data(minutes=150)

def creation_tuples(ids, path = 'data_traite\\p3d{}.csv'):
    return [(id, path.format(id)) for id in ids]

ids = [1, 4, 10, 12, 13, 14, 16, 17, 19, 20, 22, 24, 25, 26, 27, 29, 31, 34, 38, 41, 43, 44, 46, 49, 50, 52, 53, 55, 56, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 74, 75, 77, 79, 83, 84, 87, 89, 92, 94, 96, 98, 101, 103, 104, 105, 108, 110, 111, 112, 113, 114, 116, 117, 118, 125, 126, 128, 130, 132, 135, 136, 137, 138, 139, 140, 142, 143, 145, 146, 148, 149, 150, 152, 153, 156, 160, 161, 163, 164, 166, 167, 168, 172, 175, 177, 178, 181, 183, 186, 190, 191, 195, 198, 199,200, 202, 203, 206, 207, 210, 218, 221, 222, 232, 233, 234, 236, 237, 239, 241, 244, 247, 250, 251, 252, 256, 258, 261, 263, 264, 266, 269, 270, 279, 280, 281, 282, 283, 284, 286, 287, 293, 295, 296, 297, 300, 303, 304, 306, 308, 309, 312, 316, 318, 319, 321, 323, 325, 326, 330, 337, 338, 342, 343, 345, 347, 348, 349, 353, 355, 357, 358, 359, 362, 363, 366, 367, 369, 371, 375, 380, 382, 383, 384, 387, 388, 390, 395, 397, 398, 402, 404, 405, 406, 408, 
409, 413, 416, 417, 418, 419, 424, 425, 431, 435, 438, 439, 440, 441, 442, 445, 447, 448, 451, 452, 455, 458, 462, 466, 469, 472, 474, 476, 478, 483, 484, 485, 486, 488, 489, 492, 495, 505, 506, 507, 512, 513, 514, 516, 519, 521, 526, 530, 531, 533, 537, 541, 543, 547, 550, 551, 553, 559, 560, 561, 562, 563, 564, 565, 566, 568, 570, 573, 575, 578, 579, 582, 584, 590, 593, 594, 599, 1292,1730,1820,1900,1959,2327,2332,2738,3113,3188,3270,3524,3719,3930,4255,5304,5607,5682,5884,5983,6009,6227]

tuples = creation_tuples(ids)

def save_data(data, filename):
    nom_colonnes = ['id','index','labelpre', 'labeltarget1', 'labeltarget2', 'labeltarget3','labeltarget4','labeltarget5', 'labeltarget']
    try:
        if os.path.exists(filename):
            os.remove(filename)
        if data is None:
            print("Aucune data founie")
            return
        if isinstance(data, pd.DataFrame):
            data.to_csv(filename, index=False)
            print(f"Données sauvegardées dans {filename}")
        elif isinstance(data, list):
            if len(data) == 0:
                raise ValueError("la liste de données est vide.")
            else:
                nb_colonnes = len(data[0])
                nom_colonnes_totale = [f"Col{i}" for i in range(0, nb_colonnes - len(nom_colonnes))] + nom_colonnes
                df = pd.DataFrame(data, columns=nom_colonnes_totale)
                df.to_csv(filename, index=False)
                print(f"Données listées sauvegardées dans {filename}.")
        else:
            print("format des donées n'est ni dataframe ni liste.")
    except Exception as e:
        print(f"Erreur lors de sauvegarde des données : {e}")

def get_data(id, filename):
    try:
        data = caracteristique.generate(id)
        if data is None:
            print(f"Traitement non réussie pour patient {id} : non sauvegardé dans {filename} à cause que indicateur retourne des valeurs none.")
        else:
            save_data(data, filename)
            print(f"Traitement réussie pour patient {id} : sauvegardé dans {filename}.")
    except Exception as e:
        print(f"Erreur pour echantillon {id} : {e}")



if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data_features = [executor.submit(get_data, id, filename) for id, filename in tuples]
        for data_feature in concurrent.futures.as_completed(data_features):
            data_feature.result()