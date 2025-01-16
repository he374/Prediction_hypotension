import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

os.environ['TCL_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\Admin\AppData\Local\Programs\Python\Python313\tcl\tk8.6'

map = "Solar8000/ART_MBP"
bis = "BIS/BIS"
dap = "Solar8000/ART_DBP"
sap = "Solar8000/ART_SBP"

class DATA_anesthesie:

    def __init__(self, id):
        self.m = pd.read_csv(f"data_vb\\map\\map{id}.csv")
        self.b = pd.read_csv(f"data_vb\\bis\\bis{id}.csv")
        self.s = pd.read_csv(f"data_vb\\sap\\sap{id}.csv")
        self.d = pd.read_csv(f"data_vb\\dap\\dap{id}.csv")

    def indicateur(self):
        datam = self.m
        datab = self.b
        if datab is None or datam is None:
            return None
        else:
            datab = datab[datab[bis]!=0].reset_index(drop=True)
            debut = None
            fin = None
            for i in range(1, len(datab)):
                if datab[bis].iloc[i-1]>60>=datab[bis].iloc[i]:
                    debut = i
                    break
            for i in range(len(datab)-1,0, -1):
                if datab[bis].iloc[i-1] < 60 <= datab[bis].iloc[i]:
                    fin = i
                    break

            return debut, fin
        
    @staticmethod
    def numero_ech(debut,fin):
        if debut is None and fin is not None:
            debut = 0
        elif fin is None:
            return None
        else:
            s = 2.5 * (fin-debut)/150
            sm = max(0, int(s)-1)
            return sm

    def echantillonnage_predata(self,parametre, index, fenetre_pas, longeur_fenetre):
        debut, fin = self.indicateur()
        debut_fenetre = index*fenetre_pas + debut
        data_sampled = []
        if parametre == "map":
            data = self.m
        elif parametre == "bis":
            data = self.b
        elif parametre == "sap":
            data = self.s
        elif parametre == "dap":
            data = self.d
        
        if data is None or debut is None or fenetre_pas is None or longeur_fenetre is None:
            print(f"erreur fonction echantillonage de predata de parametre {parametre} et id {id} et index {index} : data None")
        else:
            for i in range(debut_fenetre, debut_fenetre + longeur_fenetre):
                try:
                    data_sampled.append(data.iloc[i].to_dict())
                except IndexError:
                    print(f"indice {i} hors limites pour les données")
                    break

            if len(data_sampled) > 0:
                return pd.DataFrame(data_sampled)
            else:
                print(f"Predata non colectée de id {id} et indice {index}")
                return None
            
    def echantillonnage_cumulative_predata(self, parametre, index, fenetre_pas, longeur_fenetre):
        debut, fin = self.indicateur()
        debut_fenetre = index*fenetre_pas + debut
        data_sampled = []
        if parametre == "map":
            data = self.m
        elif parametre == "bis":
            data = self.b
        elif parametre == "sap":
            data = self.s
        elif parametre == "dap":
            data = self.d
        if data is None or debut is None or fenetre_pas is None or longeur_fenetre is None:
            print(f"erreur fonction echantillonage de predata de parametre {parametre} et id {id} et index {index} : data None")
        else:
            for i in range(debut, debut_fenetre + longeur_fenetre):
                try:
                    data_sampled.append(data.iloc[i].to_dict())
                except IndexError:
                    print(f"indice {i} hors limites pour les données")
                    break

            if len(data_sampled) > 0:
                return pd.DataFrame(data_sampled)
            else:
                print(f"Predata non colectée de id {id} et indice {index}")
                return None
            
    def echantillonnage_targetdata(self, parametre, index, fenetre_pas, longeur_fenetre, longeur_fenetre_predata):
        debut, fin = self.indicateur()
        debut_fenetre = index*fenetre_pas + debut + longeur_fenetre_predata
        data_sampled = []
        if parametre == "map":
            data = self.m
        elif parametre == "bis":
            data = self.b
        elif parametre == "sap":
            data = self.s
        elif parametre == "dap":
            data = self.d
        if data is None or debut is None or fenetre_pas is None or longeur_fenetre is None:
            print(f"erreur fonction echantillonage de predata de parametre {parametre} et id {id} et index {index} : data None")
        else:
            for i in range(debut_fenetre, debut_fenetre + longeur_fenetre):
                try:
                    data_sampled.append(data.iloc[i].to_dict())
                except IndexError:
                    print(f"indice {i} hors limites pour les données")
                    break

            if len(data_sampled) > 0:
                return pd.DataFrame(data_sampled)
            else:
                print(f"Predata non colectée de id {id} et indice {index}")
                return None

    
class features():
    def __init__(self, id, index, minutes):
        self.id = id
        self.index = index
        self.minutes = minutes
        DA = DATA_anesthesie(id)
        self.predatam = DA.echantillonnage_predata("map", index, fenetre_pas=60, longeur_fenetre=minutes)
        self.predatab = DA.echantillonnage_predata("bis", index, fenetre_pas=60, longeur_fenetre=minutes)
        self.predatas = DA.echantillonnage_predata("sap",index, fenetre_pas=60, longeur_fenetre=minutes)
        self.predatad = DA.echantillonnage_predata("dap", index, fenetre_pas=60, longeur_fenetre=minutes)
        self.cumpredatam = DA.echantillonnage_cumulative_predata("map", index, fenetre_pas=60, longeur_fenetre=minutes)
        self.targetdata1 = DA.echantillonnage_targetdata("map", index, fenetre_pas=60, longeur_fenetre=60, longeur_fenetre_predata=minutes)
        self.targetdata2 = DA.echantillonnage_targetdata("map", index+1, fenetre_pas=60, longeur_fenetre=60, longeur_fenetre_predata=minutes)
        self.targetdata3 = DA.echantillonnage_targetdata("map", index+2, fenetre_pas=60, longeur_fenetre=60, longeur_fenetre_predata=minutes)
        self.targetdata4 = DA.echantillonnage_targetdata("map", index+3, fenetre_pas=60, longeur_fenetre=60, longeur_fenetre_predata=minutes)
        self.targetdata5 = DA.echantillonnage_targetdata("map", index+4, fenetre_pas=60, longeur_fenetre=60, longeur_fenetre_predata=minutes)

    def labels(self):
        l = []
        def detect_hypotension(data):
            seuil = 65
            largeur_window = 30
            if data is None:
                return None
            else:
                df = data[map]
                for i in range(len(data) - largeur_window +1):
                    if all(float(value)<seuil for value in df.iloc[i:i+largeur_window]):
                        return True
                return False
        label_pre = detect_hypotension(self.predatam)
        label1 = detect_hypotension(self.targetdata1)
        label2 = detect_hypotension(self.targetdata2)
        label3 = detect_hypotension(self.targetdata3)
        label4 = detect_hypotension(self.targetdata4)
        label5 = detect_hypotension(self.targetdata5)
        label = label1 | label2 | label3 | label4 | label5
        return [label_pre, label1, label2, label3, label4, label5, label]
    
    def liste_temps_reel_hypotension(self, parametre, seuil, largeur_window):
        l = []
        if parametre == "map":
            parametre = map
            data = self.predatam
        elif parametre == "bis":
            parametre = bis
            data = self.predatab
        elif parametre == "sap":
            parametre = sap
            data = self.predatas
        elif parametre == "dap":
            parametre = dap
            data = self.predatad
        
        if data is None:
            return None
        else:
            df = data[parametre]
            for i in range(len(data)-largeur_window+1):
                if all(float(value)<seuil for value in df.iloc[i:i+largeur_window]):
                    l.append(2)
                elif any(float(value)<=seuil for value in df.iloc[i:i+largeur_window]) and not all(float(value)<seuil for value in df.iloc[i:i+largeur_window]):
                    l.append(1)
                else:
                    l.append(0)
            return l
        
    def liste_temps_reel_hypotension_v2(self, parametre, seuil1, seuil2, largeur_window):
        l = []
        if parametre == "map":
            parametre = map
            data = self.predatam
        elif parametre == "bis":
            parametre = bis
            data = self.predatab
        elif parametre == "sap":
            parametre = sap
            data = self.predatas
        elif parametre == "dap":
            parametre = dap
            data = self.predatad
        
        if data is None:
            return None
        else:
            df = data[parametre]
            for i in range(len(data)-largeur_window+1):
                if all(float(value)<seuil1 for value in df.iloc[i:i+largeur_window]):
                    l.append(4)
                elif all(float(value)<seuil2 for value in df.iloc[i:i+largeur_window]):
                    l.append(5)
                elif any(float(value)<=seuil1 for value in df.iloc[i:i+largeur_window]) and not all(float(value)<seuil1 for value in df.iloc[i:i+largeur_window]):
                    l.append(2)
                elif any(float(value)<=seuil2 for value in df.iloc[i:i+largeur_window]) and not all(float(value)<seuil2 for value in df.iloc[i:i+largeur_window]):
                    l.append(3)
                elif all(float(value)>=seuil1 for value in df.iloc[i:i+largeur_window]):
                    l.append(1)
                else:
                    l.append(0)
            return l
        
    @staticmethod
    def compter_occurrences(l):
        r = []
        if len(l)!=0:
            ph = l.count(2)/len(l)
            pd = l.count(1)/len(l)
            r.append(ph)
            r.append(pd)
            return r
        else:
            return None
        
    def puissance(self):
        l = []
        def detect_hypotension_episodes(data, seuil = 65, duration = 30):
            episodes = []
            inf_seuil = data[map]<seuil
            i = 0
            while i < len(inf_seuil):
                if inf_seuil.iloc[i]:
                    debut = i
                    while i < len(inf_seuil) and inf_seuil.iloc[i]:
                        i += 1
                    fin = i
                    if (fin-debut) >= duration:
                        episodes.append((debut, fin))
                i+=1
            return episodes
        def calculer(map_signal, episodes):
            n = len(map_signal)
            p_m = np.sum(np.square(map_signal))/n
            p_h_i = []
            for debut, fin in episodes:
                p_i = np.sum(np.square(map_signal[debut:fin]))/(fin-debut)
                p_h_i.append(p_i)
            n_h = len(p_h_i)
            p_h = np.mean(p_h_i) if n_h >0 else 0
            r = p_h /p_m if p_m != 0 else 0

            l.append(p_m)
            l.append(p_h)
            l.append(r)
            return l
        data = self.predatam
        if data is not None:
            episodes = detect_hypotension_episodes(data)
            result = calculer(data[map], episodes)
            return result
        else:
            return None
        
    def parametre_statistique(self):
        datam = self.predatam
        cumdatam = self.cumpredatam
        seuil = 65
        fs = 0.5
        n = self.minutes//50
        var1 = len(datam)//n
        var1 = int(var1)
        n = int(n)
        all = []
        l = []
        for i in range(n):
            i1 = i*var1
            i2 = (i+1)*var1 if i<n else len(datam)
            d = datam.iloc[i1:i2]
            cm = cumdatam.iloc[i1:i2]
            map_signal = d[map]
            map_signal.reset_index(drop=True, inplace=True)
            if map_signal is not None:
                valeur_moyenne1=map_signal.mean()
                map_signal= map_signal.fillna(valeur_moyenne1)
            map_signalcum = cm[map]
            map_signalcum.reset_index(drop=True, inplace=True)            
            if map_signalcum is not None:
                valeur_moyenne2=map_signalcum.mean()
                map_signalcum= map_signal.fillna(valeur_moyenne2)
            if d is None or cm is None:
                print("predata ou predatacumulative est vide")
                return None
            if map_signal is None:
                l.append(None)
                l.append(None)
                l.append(None)
                l.append(None)
                l.append(None)
            elif len(map_signal)<2:
                l.append(None)
                l.append(None)
                l.append(None)
                l.append(None)
                l.append(None)
            else:
                l = []
                def delta():
                    diffs = np.abs(np.diff(map_signal))
                    return np.mean(diffs)
                def central_tendency_measure():
                    count = 0
                    variations= np.diff(map_signal)
                    rho = 1.5*np.std(variations)
                    for i in range(len(map_signal)-2):
                        racine = np.sqrt((map_signal[i+2]-map_signal[i+1])**2+(map_signal[i+1]-map_signal[i])**2)
                        if racine < rho:
                            count+=1
                    return count/(len(map_signal)-2)
                def cumulative_time_below_threshold():
                    if len(map_signalcum) == 0:
                        return 0
                    else:
                        sous_seuil = np.sum(map_signalcum<seuil)
                        return 100*sous_seuil/len(map_signalcum)
                def approximative_entropy(m = 2):
                    def phi(m):
                        N=len(map_signal)
                        if N<=m:
                            return 0
                        std_dev = np.std(map_signal)
                        r = 0.2 * std_dev
                        patterns = [map_signal[i:i + m] for i in range(N - m + 1)]
                        patterns = np.array(patterns)
                        distances = np.abs(patterns[:, None, :] - patterns[None, :, :]).max(axis=2)
                        counts = np.sum(distances <= r, axis=0)
                        counts = np.where(counts == 0, 1, counts)
                        return np.sum(np.log(counts / (N - m + 1))) / (N - m + 1)
                    phi_m = phi( m)
                    phi_m_plus_1 = phi( m + 1)
                    return phi_m - phi_m_plus_1
                def map_hypotension_index():
                    temps_total = len(map_signalcum)/fs
                    if len(map_signalcum)<2 or temps_total == 0:
                        return 0 
                    hypotensions = 0
                    for i in range(1, len(map_signalcum)):
                        if map_signalcum[i-1]>=seuil and map_signalcum[i]<=seuil:
                            hypotensions+=1
                    return hypotensions/(temps_total/3600)
                r1 = delta()
                r2 = central_tendency_measure()
                r3 = cumulative_time_below_threshold()
                r4 = approximative_entropy()
                r5 = map_hypotension_index()
                l.append(r1)
                l.append(r2)
                l.append(r3)
                l.append(r4)
                l.append(r5)
                all.extend(l)
        return all
        
    def reg_polynomiale(self, ordre=3):
        n = self.minutes//50
        datam = self.predatam
        var1 = len(datam)//n
        var1 = int(var1)
        n = int(n)
        l = []
        for i in range(n):
            i1 = i*var1
            i2 = (i+1)*var1 if i<n else len(datam)
            d = datam.iloc[i1:i2]
            if d[map].isnull().any():
                valeur_moyenne = d[map].mean()
                d[map]=d[map].fillna(valeur_moyenne)
            x = d['Time'].values
            x2 = x.reshape(-1,1)
            y = d[map].values
            poly = PolynomialFeatures(degree=ordre)
            x_poly = poly.fit_transform(x2)
            model = LinearRegression()
            model.fit(x_poly, y)
            coffs = model.coef_
            intercept = model.intercept_
            l.extend(coffs)
            l.append(intercept)
        return l
                
