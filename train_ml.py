import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import mean_squared_error, accuracy_score,classification_report, recall_score, precision_score
from sklearn.model_selection import train_test_split , cross_val_score, KFold,LeaveOneOut, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




def tester_classifications(data,id_col,  last_col):
    
    classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),  
    "ExtraTrees": ExtraTreesClassifier(),
    "Linear Discriminant Analysis": LDA(),
    "Support Vector Machine": SVC(probability=True),
    "Quadratic Discriminant Analysis": QDA(),
    "MLP": MLPClassifier(max_iter=1000)  
    }
    scaler = StandardScaler()
    unique_ids = data[id_col].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)  
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  
    assert not set(train_ids) & set(val_ids), "Fuite détectée entre train et validation"
    assert not set(train_ids) & set(test_ids), "Fuite détectée entre train et test"
    assert not set(val_ids) & set(test_ids), "Fuite détectée entre validation et test"
    train_data = data[data[id_col].isin(train_ids)]
    val_data = data[data[id_col].isin(val_ids)]
    test_data = data[data[id_col].isin(test_ids)]
    class_minority = train_data[train_data[last_col] == True]
    class_majority = train_data[train_data[last_col] == False]

    class_minority_oversampled = resample(
        class_minority, replace=True, n_samples=len(class_majority), random_state=42
    )
    data_balanced = pd.concat([class_majority, class_minority_oversampled])

    x_train = data_balanced.drop(columns=[last_col, id_col])
    y_train = data_balanced[last_col].map({False: 0, True: 1}).astype(int)

    x_val = val_data.drop(columns=[last_col, id_col])
    y_val = val_data[last_col].map({False: 0, True: 1}).astype(int)

    x_test = test_data.drop(columns=[last_col, id_col])
    y_test = test_data[last_col].map({False: 0, True: 1}).astype(int)

    print(y_train.unique())  
    print(y_train.dtype) 
    print(y_train.value_counts())
    print(x_train.shape )
    print(x_test.shape)
    print(data_balanced)

    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

        

    for name, model in classifiers.items():
        print(f"\n{name}")
        model.fit(x_train_scaled, y_train)
        val_predictions = model.predict(x_val_scaled)
        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {val_accuracy:.2f}")
        test_predictions = model.predict(x_test_scaled)
        test_accuracy = accuracy_score(y_test, test_predictions)
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print(classification_report(y_test, test_predictions))



def train_meta_model(data, last_col, id_col, outer_splits=5, inner_splits=3):
    unique_ids = data[id_col].unique()
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)  
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  

    assert not set(train_ids) & set(val_ids), "Fuite détectée entre train et validation"
    assert not set(train_ids) & set(test_ids), "Fuite détectée entre train et test"
    assert not set(val_ids) & set(test_ids), "Fuite détectée entre validation et test"

    train_data = data[data[id_col].isin(train_ids)]
    val_data = data[data[id_col].isin(val_ids)]
    test_data = data[data[id_col].isin(test_ids)]

    class_minority = train_data[train_data[last_col] == True]
    class_majority = train_data[train_data[last_col] == False]

    class_minority_oversampled = resample(
        class_minority, replace=True, n_samples=len(class_majority), random_state=42
    )
    data_balanced = pd.concat([class_majority, class_minority_oversampled])

    x_train = data_balanced.drop(columns=[last_col, id_col])
    y_train = data_balanced[last_col].map({False: 0, True: 1}).astype(int)

    x_val = val_data.drop(columns=[last_col, id_col])
    y_val = val_data[last_col].map({False: 0, True: 1}).astype(int)

    x_test = test_data.drop(columns=[last_col, id_col])
    y_test = test_data[last_col].map({False: 0, True: 1}).astype(int)

    base_learners = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('et', ExtraTreesClassifier(random_state=42)),
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('knn', KNeighborsClassifier()),
        ('mlp', MLPClassifier(random_state=42, max_iter=1000)),
        ('lda', LDA()),
        ('qda', QDA()),
        ('rdn', RidgeClassifier()),
        ('log', LogisticRegression(random_state=42))
    ]

    meta_model = QDA()

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=42)

    final_predictions = []
    true_labels = []

    for train_idx, val_idx in outer_cv.split(x_train, y_train):
        x_train_outer, x_val_outer = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_train_outer, y_val_outer = y_train.iloc[train_idx], y_train.iloc[val_idx]

        stacking_clf = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_model,
            cv=inner_cv
        )

        stacking_clf.fit(x_train_outer, y_train_outer)
        y_val_pred = stacking_clf.predict(x_val_outer)
        final_predictions.extend(y_val_pred)
        true_labels.extend(y_val_outer)

    stacking_clf.fit(x_train, y_train)
    y_test_pred = stacking_clf.predict(x_test)

    print("\nClassification Report (Validation externe NTTV) :")
    print(classification_report(true_labels, final_predictions))

    print("\nClassification Report (Test final) :")
    print(classification_report(y_test, y_test_pred))


def plot_save_training(name, model):
    p = []
    r = []
    num_rows = []
    id_col = 'id'
    l = ["1er 2m","2eme 2m","3eme 2m","4eme 2m","5eme 2m","8 m","10 m"]
    d = ["cd1.csv","cd2.csv","cd3.csv","cd4.csv","cd5.csv","cd8.csv","cd.csv"]
    labels = ['labeltarget1', 'labeltarget2', 'labeltarget3', 'labeltarget4' , 'labeltarget5' , 'labeltarget8' , 'labeltarget' ]
    for i in range(len(d)):
        data = pd.read_csv(d[i], index_col=False)
        last_col = labels[i]
        scaler = StandardScaler()
        unique_ids = data[id_col].unique()
        train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)  
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)  

        assert not set(train_ids) & set(val_ids), "Fuite détectée entre train et validation"
        assert not set(train_ids) & set(test_ids), "Fuite détectée entre train et test"
        assert not set(val_ids) & set(test_ids), "Fuite détectée entre validation et test"

        train_data = data[data[id_col].isin(train_ids)]
        val_data = data[data[id_col].isin(val_ids)]
        test_data = data[data[id_col].isin(test_ids)]
        class_minority = train_data[train_data[last_col] == True]
        class_majority = train_data[train_data[last_col] == False]

        class_minority_oversampled = resample(
            class_minority, replace=True, n_samples=len(class_majority), random_state=42
        )
        data_balanced = pd.concat([class_majority, class_minority_oversampled])

        x_train = data_balanced.drop(columns=[last_col, id_col])
        y_train = data_balanced[last_col].map({False: 0, True: 1}).astype(int)

        x_val = val_data.drop(columns=[last_col, id_col])
        y_val = val_data[last_col].map({False: 0, True: 1}).astype(int)

        x_test = test_data.drop(columns=[last_col, id_col])
        y_test = test_data[last_col].map({False: 0, True: 1}).astype(int)

        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)
        

            


        
        if model == QDA():
            pca = PCA(n_components=10)
            X_train_pca = pca.fit_transform(x_train_scaled)
            X_test_pca = pca.fit_transform(x_test_scaled)
            model.fit(X_train_pca, y_train)
            test_predictions = model.predict(X_test_pca)
            recall = recall_score(y_test, test_predictions)
            precision = precision_score(y_test, test_predictions)
            num_rows.append(l[i])
            p.append(precision)
            r.append(recall)
        else:
            model.fit(x_train_scaled, y_train)
            test_predictions = model.predict(x_test_scaled)
            recall = recall_score(y_test, test_predictions)
            precision = precision_score(y_test, test_predictions)
            num_rows.append(l[i])
            p.append(precision)
            r.append(recall)
    plt.figure(figsize=(10, 6))
    plt.plot(num_rows, p, label='Précision', linestyle='-', marker='.')
    plt.plot(num_rows, r, label='Rappel', linestyle='--', marker='.')
    plt.title(f'Précision et Rappel - {name}', fontsize=14)
    plt.xlabel('target data', fontsize=12)
    plt.ylabel('Valeur', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(f'models_performance/{name}_precision_recall.png')
    plt.close()

def save_plots():
    voting_clf = VotingClassifier(
        estimators=[
            ('Logistic Regression', LogisticRegression(max_iter=5000)),
            ('Random Forest', RandomForestClassifier()),
            ("Quadratic Discriminant Analysis", QDA())
        ],
        voting='soft'
    )
    from sklearn.calibration import CalibratedClassifierCV
    logreg = LogisticRegression()
    rf = RandomForestClassifier()
    calibrated_rf1 = CalibratedClassifierCV(estimator=rf, method='sigmoid')
    calibrated_rf2 = CalibratedClassifierCV(estimator=logreg, method='sigmoid')
    calibrated_rf3 = CalibratedClassifierCV(estimator=rf, method='isotonic')
    calibrated_rf4 = CalibratedClassifierCV(estimator=logreg, method='isotonic')
    classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),  
    "ExtraTrees": ExtraTreesClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Linear Discriminant Analysis": LDA(),
    "Quadratic Discriminant Analysis": QDA(),
    "calibrated_rf2":     calibrated_rf2,
    "calibrated_rf1":  calibrated_rf1,
    "calibrated_rf3":  calibrated_rf3,
    "calibrated_rf4":  calibrated_rf4,
    "voting_clf":     voting_clf,
    "MLP": MLPClassifier(max_iter=1000) 
    }   

    for name, model in classifiers.items():
        plot_save_training(name, model)


def tester_voting_clf():
    p = []
    r = []
    num_rows = []
    id_col = 'id'
    l = ["1er 2m","2eme 2m","3eme 2m","4eme 2m","5eme 2m","8 m","10 m"]
    d = ["cd1.csv","cd2.csv","cd3.csv","cd4.csv","cd5.csv","cd8.csv","cd.csv"]
    labels = ['labeltarget1', 'labeltarget2', 'labeltarget3', 'labeltarget4' , 'labeltarget5' , 'labeltarget8' , 'labeltarget' ]
    for i in range(len(d)):
        last_col = labels[i]
        data = pd.read_csv(d[i], index_col=False)

        scaler = StandardScaler()
        unique_ids = data[id_col].unique()
        train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
        assert not set(train_ids) & set(val_ids), "Fuite détectée entre train et validation"
        assert not set(train_ids) & set(test_ids), "Fuite détectée entre train et test"
        assert not set(val_ids) & set(test_ids), "Fuite détectée entre validation et test"

        train_data = data[data[id_col].isin(train_ids)]
        val_data = data[data[id_col].isin(val_ids)]
        test_data = data[data[id_col].isin(test_ids)]

        class_minority = train_data[train_data[last_col] == True]
        class_majority = train_data[train_data[last_col] == False]

        class_minority_oversampled = resample(
            class_minority, replace=True, n_samples=len(class_majority), random_state=42
        )
        data_balanced = pd.concat([class_majority, class_minority_oversampled])

        x_train = data_balanced.drop(columns=[last_col, id_col])
        y_train = data_balanced[last_col].map({False: 0, True: 1}).astype(int)

        x_val = val_data.drop(columns=[last_col, id_col])
        y_val = val_data[last_col].map({False: 0, True: 1}).astype(int)

        x_test = test_data.drop(columns=[last_col, id_col])
        y_test = test_data[last_col].map({False: 0, True: 1}).astype(int)

        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)

        # Ajouter le vote majoritaire
        voting_clf = VotingClassifier(
            estimators=[
                ('Logistic Regression', LogisticRegression(max_iter=5000)),
                ('Random Forest', RandomForestClassifier()),
                ("Quadratic Discriminant Analysis", QDA())
            ],
            voting='soft'
        )
        voting_clf.fit(x_train_scaled, y_train)
        y_val_voting = voting_clf.predict(x_val_scaled)
        y_test_voting = voting_clf.predict(x_test_scaled)
        recall = recall_score(y_test, y_test_voting)
        precision = precision_score(y_test, y_test_voting)
        num_rows.append(l[i])
        p.append(precision)
        r.append(recall)    
    plt.figure(figsize=(10, 6))
    plt.plot(num_rows, p, label='Précision', linestyle='-', marker='.')
    plt.plot(num_rows, r, label='Rappel', linestyle='--', marker='.')
    plt.title('Précision et Rappel - voting clf', fontsize=14)
    plt.xlabel('target data', fontsize=12)
    plt.ylabel('Valeur', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig('votingclf_performance/voting clf_precision_recall.png')
    plt.close()



def tester_stacking_clf():
    p = []
    r = []
    num_rows = []
    id_col = 'id'
    l = ["1er 2m", "2eme 2m", "3eme 2m", "4eme 2m", "5eme 2m", "8 m", "10 m"]
    d = ["cd1.csv", "cd2.csv", "cd3.csv", "cd4.csv", "cd5.csv", "cd8.csv", "cd.csv"]
    labels = ['labeltarget1', 'labeltarget2', 'labeltarget3', 'labeltarget4', 'labeltarget5', 'labeltarget8', 'labeltarget']

    for i in range(len(d)):
        last_col = labels[i]
        data = pd.read_csv(d[i], index_col=False)

        scaler = StandardScaler()
        unique_ids = data[id_col].unique()
        train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
        assert not set(train_ids) & set(val_ids), "Fuite détectée entre train et validation"
        assert not set(train_ids) & set(test_ids), "Fuite détectée entre train et test"
        assert not set(val_ids) & set(test_ids), "Fuite détectée entre validation et test"
        train_data = data[data[id_col].isin(train_ids)]
        val_data = data[data[id_col].isin(val_ids)]
        test_data = data[data[id_col].isin(test_ids)]
        class_minority = train_data[train_data[last_col] == True]
        class_majority = train_data[train_data[last_col] == False]

        class_minority_oversampled = resample(
            class_minority, replace=True, n_samples=len(class_majority), random_state=42
        )
        data_balanced = pd.concat([class_majority, class_minority_oversampled])

        x_train = data_balanced.drop(columns=[last_col, id_col])
        y_train = data_balanced[last_col].map({False: 0, True: 1}).astype(int)

        x_val = val_data.drop(columns=[last_col, id_col])
        y_val = val_data[last_col].map({False: 0, True: 1}).astype(int)

        x_test = test_data.drop(columns=[last_col, id_col])
        y_test = test_data[last_col].map({False: 0, True: 1}).astype(int)

        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled = scaler.transform(x_val)
        x_test_scaled = scaler.transform(x_test)
        stacking_clf = StackingClassifier(
            estimators=[
                ('Logistic Regression', LogisticRegression(max_iter=5000)),
                ('QDA', QDA())
            ],
            final_estimator=SVC(), cv=5)
        stacking_clf.fit(x_train_scaled, y_train)
        y_val_stacking = stacking_clf.predict(x_val_scaled)
        y_test_stacking = stacking_clf.predict(x_test_scaled)

        recall = recall_score(y_test, y_test_stacking)
        precision = precision_score(y_test, y_test_stacking)
        num_rows.append(l[i])
        p.append(precision)
        r.append(recall)
    plt.figure(figsize=(10, 6))
    plt.plot(num_rows, p, label='Précision', linestyle='-', marker='.')
    plt.plot(num_rows, r, label='Rappel', linestyle='--', marker='.')
    plt.title('Précision et Rappel - Stacking clf', fontsize=14)
    plt.xlabel('target data', fontsize=12)
    plt.ylabel('Valeur', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig('stackingclf_performance/stacking2_SVC_precision_recall.png')
    plt.close()

"""data represente données d'entrainement du modèle avec un targetdata de 8 min et un leading time de 2 min"""
data= pd.read_csv("cd8.csv", index_col=False)
tester_classifications(data,'id','labeltarget8')