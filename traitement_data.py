import pandas as pd
import numpy as np
from functools import reduce
import ast
import os

class PreTrain:
    def __init__(self):
        pass

    @staticmethod
    def prepare():
        folder_path = "data_traite"

        dataframes = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  
                file_path = os.path.join(folder_path, file_name)  
                print(f"Lecture du fichier : {file_name}")

                df = pd.read_csv(file_path, index_col=False)
                
                dataframes.append(df)
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            print(f"Nombre total de lignes dans le DataFrame combiné : {len(combined_df)}")
        else:
            print("Aucun fichier CSV trouvé dans le dossier.")

        print(combined_df.head())

        output_path = "combined_data.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"Les données combinées ont été sauvegardées dans : {output_path}")

        def replace_and_eval(value):
            clean_value = value.replace("np.float64", "float")
            return eval(clean_value, {"__builtins__": None}, {"float": float, "np": np})


        def convert_to_numpy(value):
            if isinstance(value, list):
                # Parcours récursif pour convertir chaque élément
                return [[np.float64(x) for x in sublist] for sublist in value]
            return value


        data = pd.read_csv("combined_data.csv")

        data = data[data['Col36'].notna()]
        data = data[data['Col37'].notna()]
        data = data[data['Col38'].notna()]

        data['Col0'] = data['Col0'].apply(ast.literal_eval)
        data['Col0'] = data['Col0'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_liste", data['Col0'].apply(len) )
        data = data[data["longeur_liste"] == 121]

        data['Col39'] = data['Col39'].apply(ast.literal_eval)
        data['Col39'] = data['Col39'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeL2", data['Col29'].apply(len) )
        data = data[data["longeur_listeL2"] == 121]

        data['Col36'] = data['Col36'].apply(ast.literal_eval)
        data['Col36'] = data['Col36'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeBIS", data['Col26'].apply(len) )
        data = data[data["longeur_listeBIS"] == 121]

        data['Col37'] = data['Col37'].apply(ast.literal_eval)
        data['Col37'] = data['Col37'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeSAP", data['Col27'].apply(len) )
        data = data[data["longeur_listeSAP"] == 121]

        data['Col38'] = data['Col38'].apply(ast.literal_eval)
        data['Col38'] = data['Col38'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeDAP", data['Col38'].apply(len) )
        data = data[data["longeur_listeDAP"] == 121]

        data['Col40'] = data['Col40'].apply(ast.literal_eval)
        data['Col40'] = data['Col40'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeBIS2", data['Col40'].apply(len) )
        data = data[data["longeur_listeBIS2"] == 121]

        data['Col41'] = data['Col41'].apply(ast.literal_eval)
        data['Col41'] = data['Col41'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeSAP2", data['Col41'].apply(len) )
        data = data[data["longeur_listeSAP2"] == 121]

        data['Col42'] = data['Col42'].apply(ast.literal_eval)
        data['Col42'] = data['Col42'].apply(lambda x: [float(i) for i in x])
        data.insert(1, "longeur_listeDAP2", data['Col42'].apply(len) )
        data = data[data["longeur_listeDAP2"] == 121]


        data = data.rename(columns={"Col0": "MAP","Col1": "PH MAP","Col2": "PD MAP","Col3": "puissance moy","Col4": "puissance hypotension","Col5": "R","Col6": "delta1","Col7": "ctm1","Col8": "CTX1","Col9": "APEN1","Col10": "PHI1","Col11": "delta2","Col12": "ctm2","Col13": "CTX2","Col14": "APEN2","Col15": "PHI2","Col16": "delta3","Col17": "ctm3","Col18": "CTX3","Col19": "APEN3","Col20": "PHI3","Col21": "delta4","Col22": "ctm4","Col23": "CTX4","Col24": "APEN4","Col25": "PHI4","Col26": "delta5","Col27": "ctm5","Col28": "CTX5","Col29": "APEN5","Col30": "PHI5","Col31": "c1","Col32": "c2","Col33": "c3","Col34": "c4","Col35": "i5","Col36": "c6","Col37": "c7","Col38": "c8","Col39": "c9","Col40": "i10","Col41": "c11","Col42": "c12","Col43": "c13","Col44": "c14","Col45": "i15","Col46": "BIS","Col47": "SAP","Col48": "DAP","Col49": "LMAP2","Col50": "LBIS2","Col51": "LSAP2","Col52": "LDAP2"})


        matrice = np.zeros((121, 2))
        matrice[:59, 0] = 1
        matrice[60:120, 1] = 2



        matrice1 = [1, 0]
        matrice2 = [0, 1]




        def m_matrice(valeur, matrice):
            return (np.dot(valeur, matrice)).tolist()
            
                    
        def m3_matrice(valeur1, valeur2, valeur3):
            
            return (np.dot(np.dot(valeur1, valeur2), valeur3)).tolist()

        def safe_m_matrice(x, matrice):
            if isinstance(x, list) and len(x) > 0:  # Vérifie que `x` est une liste non vide
                try:
                    return m_matrice(x, matrice)  # Applique m_matrice
                except Exception as e:
                    print(f"Erreur lors du traitement de {x}: {e}")  # Debug en cas de problème
                    return None  # Retourne None si m_matrice échoue
            return None  # Retourne None pour les valeurs invalides

        data.insert(0, "MAP55", data["MAP"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["MAP55"], inplace=True)

        data.insert(1, "MAP1", data['MAP55'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "MAP2", data['MAP55'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "MAP552", data["LMAP2"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["MAP552"], inplace=True)

        data.insert(1, "SMAP1", data['MAP552'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "SMAP2", data['MAP552'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "BIS55", data["BIS"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["BIS55"], inplace=True)

        data.insert(1, "BIS1", data['BIS55'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "BIS2", data['BIS55'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "BIS552", data["LBIS2"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["BIS552"], inplace=True)

        data.insert(1, "SBIS1", data['BIS552'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "SBIS2", data['BIS552'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "SAP55", data["SAP"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["SAP55"], inplace=True)

        data.insert(1, "SAP1", data['SAP55'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "SAP2", data['SAP55'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "SAP552", data["LSAP2"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["SAP552"], inplace=True)

        data.insert(1, "SSAP1", data['SAP552'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "SSAP2", data['SAP552'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "DAP55", data["DAP"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["DAP55"], inplace=True)

        data.insert(1, "DAP1", data['DAP55'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "DAP2", data['DAP55'].apply(lambda x: m_matrice(x, matrice2)))

        data.insert(0, "DAP552", data["LDAP2"].apply(lambda x: safe_m_matrice(x, matrice)))

        data.dropna(subset=["DAP552"], inplace=True)

        data.insert(1, "SDAP1", data['DAP552'].apply(lambda x: m_matrice(x, matrice1)))
        data.insert(2, "SDAP2", data['DAP552'].apply(lambda x: m_matrice(x, matrice2)))

        data['labeltarget8'] = data["labeltarget2"] | data["labeltarget3"] | data["labeltarget4"] | data["labeltarget5"]


        
        data = data.drop(columns=["BIS55","SAP55","DAP55","MAP55","MAP552","LBIS2","BIS552","LSAP2","SAP552","LDAP2","DAP552","LMAP2","BIS","DAP","SAP","MAP","index","labelpre"])
        data = data.dropna()
        data = data.drop(columns=[col for col in data.columns if 'Unnamed' in col])

        data1 = data.drop(columns=["labeltarget3","labeltarget2","labeltarget","labeltarget4","labeltarget5","labeltarget8"])
        data1.to_csv('cd1.csv', index=False)

        data2 = data.drop(columns=["labeltarget3","labeltarget1","labeltarget","labeltarget4","labeltarget5","labeltarget8"])
        data2.to_csv('cd2.csv', index=False)

        data3 = data.drop(columns=["labeltarget1","labeltarget2","labeltarget","labeltarget4","labeltarget5","labeltarget8"])
        data3.to_csv('cd3.csv', index=False)

        data4 = data.drop(columns=["labeltarget3","labeltarget2","labeltarget1","labeltarget","labeltarget5","labeltarget8"])
        data4.to_csv('cd4.csv', index=False)

        data5 = data.drop(columns=["labeltarget3","labeltarget2","labeltarget1","labeltarget4","labeltarget8","labeltarget"])
        data5.to_csv('cd5.csv', index=False)

        data8 = data.drop(columns=["labeltarget3","labeltarget5","labeltarget2","labeltarget1","labeltarget4","labeltarget"])
        data8.to_csv('cd8.csv', index=False)

        data = data.drop(columns=["labeltarget3","labeltarget2","labeltarget1","labeltarget4","labeltarget5","labeltarget8"])
        data.to_csv('cd.csv', index=False)

PreTrain.prepare()