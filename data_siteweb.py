import pandas as pd 
import numpy as np
import requests
from io import StringIO
import os
import urllib3
import logging

os.environ['TCL_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
""" data_siteweb est utilisé pour stocké les données brutes dans le répertoire data_traite afin de minimiser le temps d'execution de code et faciliter l'acces """


class data_web:
    def __init__(self):
        pass

    @staticmethod
    def general_data():
        api_url = "https://api.vitaldb.net/trks"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.text
            df = pd.read_csv(StringIO(data))
            return df
        except Exception as e:
            print(f"Erreur lors de la récupération des données générales : {e}")
            return pd.DataFrame()
        
    @staticmethod
    def clinical_data():
        api_url = "https://api.vitaldb.net/cases"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.text
            df = pd.read_csv(StringIO(data))
            return df
        except Exception as e:
            print(f"Erreur lors de la récupération des données cliniques : {e}")
            return pd.DataFrame()
        
    @staticmethod
    def detect_hypotension(data):
        threshold = 65
        window_size = 30

        if data is None:
            return None
        else:    
            df = data["Solar8000/ART_MBP"]
        

            for i in range(len(data) - window_size + 1):
                if all(float(value) < threshold for value in df.iloc[i:i + window_size]):
                    return True
            return False
    
    @staticmethod
    def specific_data(id , parameter):

        data = data_web.general_data()
        if data.empty:
            print("Aucune donnée générale disponible.")
            return None
        else:  
            result = data[data['tname'] == parameter]
            tid_voulu = result[result['caseid'] == id]['tid'].values

            if len(tid_voulu) == 0:
                print(f"Aucun `tid` trouvé pour `caseid={id}` et `parameter={parameter}`.")
                return None
            else:
                svaleur_tid = tid_voulu[0]
                api_url = f'https://api.vitaldb.net/{svaleur_tid}'
                try:
                    
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    response = requests.get(api_url, verify= False)
                    response.raise_for_status()
                    data_tid = response.text
                    data = pd.read_csv(StringIO(data_tid))
                    if data['Time'].isna().any():
                        data['Time'] = 2* data.index
                        if parameter == "Solar8000/ART_MBP":
                            window_size = 50  
                            data["Solar8000/ART_MBP"] = data["Solar8000/ART_MBP"].rolling(window=window_size).mean()

                            threshold = 110
                            
                            data.loc[data["Solar8000/ART_MBP"] > threshold, "Solar8000/ART_MBP"] = np.nan

                            data["Solar8000/ART_MBP"] = data["Solar8000/ART_MBP"].interpolate()
                            data["Solar8000/ART_MBP"] = data["Solar8000/ART_MBP"].rolling(window=window_size).mean()
                        
                            
                        return data
                    else:
                        if parameter == "Solar8000/ART_MBP":
                            window_size = 50  
                            data["Solar8000/ART_MBP"] = data["Solar8000/ART_MBP"].rolling(window=window_size).mean()

                            threshold = 110
                            
                            data.loc[data["Solar8000/ART_MBP"] > threshold, "Solar8000/ART_MBP"] = np.nan

                            data["Solar8000/ART_MBP"] = data["Solar8000/ART_MBP"].interpolate()
                            data["Solar8000/ART_MBP"] = data["Solar8000/ART_MBP"].rolling(window=window_size).mean()
                       
                        return data
                except Exception as e:
                    print(f"Erreur lors de la récupération des données spécifiques : {e}")
                    return None
    
    


ids = [1, 4, 10, 12, 13, 14, 16, 17, 19, 20, 22, 24, 25, 26, 27, 29, 31, 34, 38, 41, 43, 44, 46, 49, 50, 52, 53, 55, 56, 58, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 74, 75, 77, 79, 83, 84, 87, 89, 92, 94, 96, 98, 101, 103, 104, 105, 108, 110, 111, 112, 113, 114, 116, 117, 118, 125, 126, 128, 130, 132, 135, 136, 137, 138, 139, 140, 142, 143, 145, 146, 148, 149, 150, 152, 153, 156, 160, 161, 163, 164, 166, 167, 168, 172, 175, 177, 178, 181, 183, 186, 190, 191, 195, 198, 199,200, 202, 203, 206, 207, 210, 218, 221, 222, 232, 233, 234, 236, 237, 239, 241, 244, 247, 250, 251, 252, 256, 258, 261, 263, 264, 266, 269, 270, 279, 280, 281, 282, 283, 284, 286, 287, 293, 295, 296, 297, 300, 303, 304, 306, 308, 309, 312, 316, 318, 319, 321, 323, 325, 326, 330, 337, 338, 342, 343, 345, 347, 348, 349, 353, 355, 357, 358, 359, 362, 363, 366, 367, 369, 371, 375, 380, 382, 383, 384, 387, 388, 390, 395, 397, 398, 402, 404, 405, 406, 408, 
409, 413, 416, 417, 418, 419, 424, 425, 431, 435, 438, 439, 440, 441, 442, 445, 447, 448, 451, 452, 455, 458, 462, 466, 469, 472, 474, 476, 478, 483, 484, 485, 486, 488, 489, 492, 495, 505, 506, 507, 512, 513, 514, 516, 519, 521, 526, 530, 531, 533, 537, 541, 543, 547, 550, 551, 553, 559, 560, 561, 562, 563, 564, 565, 566, 568, 570, 573, 575, 578, 579, 582, 584, 590, 593, 594, 599, 1292,1730,1820,1900,1959,2327,2332,2738,3113,3188,3270,3524,3719,3930,4255,5304,5607,5682,5884,5983,6009,6227]

map = "Solar8000/ART_MBP"
bis = "BIS/BIS"
dap = "Solar8000/ART_DBP"
sap = "Solar8000/ART_SBP"
co = "Vigileo/CO"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
metriques = {
    "map": map,
    "bis": bis,
    "dap": dap,
    "sap": sap,
    "co": co
}
for metrique in metriques.keys():
    os.makedirs(metrique, exist_ok=True)
for id in ids:
    for nom_metrique, data_metrique in metriques.items():
        try:
            data = data_web.specific_data(id, data_metrique)
            file_path = os.path.join(nom_metrique, f"{nom_metrique}{id}.csv")
            data.to_csv(file_path, index=False)
            logging.info(f"Fichier sauvegardé : {file_path}")
        except Exception as e:
            logging.error(f"erreur pour {nom_metrique} id {id} : {e}")