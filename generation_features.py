import os
os.environ['TCL_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
from data_viatldb import DATA_anesthesie, features

class caraxteristique_data:
    def __init__(self, minutes):
        self.minutes = minutes

    def generate(self, id):
        """minutes égale à 150 pour 5 min et égale à 300 pour 10 min"""
        tous_donnee_echantillons = []
        DA = DATA_anesthesie(id)
        debut, fin = DA.indicateur()
        n_echantillons = DA.numero_ech(debut, fin)
        if n_echantillons is None:
            print(f'n_echantillons de patient {id} est None')
            return None
        else:
            for index in range(0, n_echantillons):
                FT = features(id, index, self.minutes)
                donnee_echantillons = []
                print(f"Traitement de l'échantillon {index} de patient {id} sur {n_echantillons}")
                l_map = FT.liste_temps_reel_hypotension("map", 65, 30)
                if l_map is None:
                    print("liste map vide")
                    break
                l_bis = FT.liste_temps_reel_hypotension("bis", 40, 30)
                if l_bis is None:
                    print("liste bis vide")
                    break
                l_sap = FT.liste_temps_reel_hypotension("sap", 90, 30)
                if l_sap is None:
                    print("liste sap vide")
                    break
                l_dap = FT.liste_temps_reel_hypotension("dap", 40, 30)
                if l_dap is None:
                    print("liste dap vide")
                    break

                l_map2 = FT.liste_temps_reel_hypotension_v2("map", 65, 50, 30)
                if l_map2 is None:
                    print("liste map2 vide")
                    break
                l_bis2 = FT.liste_temps_reel_hypotension_v2("bis", 40, 20, 30)
                if l_bis2 is None:
                    print("liste bis2 vide")
                    break
                l_sap2 = FT.liste_temps_reel_hypotension_v2("sap", 90, 55, 30)
                if l_sap2 is None:
                    print("liste sap2 vide")
                    break
                l_dap2 = FT.liste_temps_reel_hypotension_v2("dap", 40, 25, 30)
                if l_dap2 is None:
                    print("liste dap2 vide")
                    break
                
                phd = FT.compter_occurrences(l_map)
                if phd is None:
                    print("phd vide")
                    break

                puissance = FT.puissance()
                if puissance is None:
                    print("liste de puissance vide")
                    break

                statistique_liste = FT.parametre_statistique()
                if statistique_liste is None:
                    print("liste des parametres statistiques vide")
                    break

                reg_liste = FT.reg_polynomiale()
                if reg_liste is None:
                    print("liste des parametres de regression vide")
                    break

                datalabel = FT.labels()
                if datalabel is None:
                    print("liste des labels vide")
                    break

                donnee_echantillons.append(l_map)
                donnee_echantillons.extend(phd)
                donnee_echantillons.extend(puissance)
                donnee_echantillons.extend(statistique_liste)
                donnee_echantillons.extend(reg_liste)
                donnee_echantillons.append(l_bis)
                donnee_echantillons.append(l_sap)
                donnee_echantillons.append(l_dap)
                donnee_echantillons.append(l_map2)
                donnee_echantillons.append(l_bis2)
                donnee_echantillons.append(l_sap2)
                donnee_echantillons.append(l_dap2)
                donnee_echantillons.append(id)
                donnee_echantillons.append(index)
                donnee_echantillons.extend(datalabel)
                tous_donnee_echantillons.append(donnee_echantillons)


            return tous_donnee_echantillons

        

