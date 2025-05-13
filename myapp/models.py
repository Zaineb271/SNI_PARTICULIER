from django.db import models
from imblearn.over_sampling import SMOTE, SMOTENC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
import scipy.stats as stat
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from django.conf import settings
import joblib


class ProcessedData(models.Model):
    column_name = models.CharField(max_length=255)
    old_type = models.CharField(max_length=50)
    new_type = models.CharField(max_length=50)
    data = models.JSONField()  # You can store the processed data as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    
    

def split_data(df, target, test_size=0.2, random_state=None):
    """
    Divise les données en ensembles d'entraînement et de test.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.
    target (str): Le nom de la variable cible.
    test_size (float): La proportion de l'ensemble de test (default=0.2).
    random_state (int): La graine pour le générateur de nombres aléatoires (default=None).
    
    Returns:
    tuple: Les ensembles d'entraînement et de test sous la forme de (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test


def balance_train_data(X_train, y_train, categorical_features, random_state=None):
   
    """
    Applique SMOTE pour les variables numériques et SMOTENC pour les variables catégorielles afin d'équilibrer l'ensemble d'entraînement.
    
    Parameters:
    X_train (pd.DataFrame): Les caractéristiques de l'ensemble d'entraînement.
    y_train (pd.Series): La variable cible de l'ensemble d'entraînement.
    categorical_features (list): La liste des noms des colonnes catégorielles.
    random_state (int): La graine pour le générateur de nombres aléatoires (default=None).
    
    Returns:
    tuple: Les caractéristiques et la variable cible de l'ensemble d'entraînement équilibré sous la forme de (X_train_balanced, y_train_balanced).
    """
    # Identifier les indices des colonnes catégorielles
    cat_indices = [X_train.columns.get_loc(col) for col in categorical_features if col in X_train.columns]

    # Vérifier si les indices catégoriels sont fournis correctement
    if not cat_indices:
        raise ValueError("Les colonnes catégorielles spécifiées ne correspondent pas aux colonnes de X_train.")
    
    # Appliquer SMOTENC pour les variables catégorielles
    smotenc = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    X_train_balanced, y_train_balanced = smotenc.fit_resample(X_train, y_train)
    
    return X_train_balanced, y_train_balanced



def encode_categorical_features(X_train,X_test, categorical_features):
    """
    Encode les variables catégorielles en utilisant OneHotEncoder.
    
    Parameters:
    X_train (pd.DataFrame): Les caractéristiques de l'ensemble d'entraînement.
    X_test (pd.DataFrame): Les caractéristiques de l'ensemble de test.
    categorical_features (list): La liste des noms des colonnes catégorielles.
    
    Returns:
    tuple: Les caractéristiques encodées de l'ensemble d'entraînement et de test.
    """
    # Initialiser l'encodeur
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Appliquer l'encodage sur les colonnes catégorielles de l'ensemble d'entraînement
    X_train_encoded = encoder.fit_transform(X_train[categorical_features])
    
    # Appliquer l'encodage sur les colonnes catégorielles de l'ensemble de test
    X_test_encoded = encoder.transform(X_test[categorical_features])

    # Convertir les résultats encodés en DataFrame
    encoded_train_df = pd.DataFrame(X_train_encoded, columns=encoder.get_feature_names_out(categorical_features), index=X_train.index)
    encoded_test_df = pd.DataFrame(X_test_encoded, columns=encoder.get_feature_names_out(categorical_features), index=X_test.index)

    # Supprimer les colonnes catégorielles originales et ajouter les colonnes encodées
    X_train = X_train.drop(columns=categorical_features).join(encoded_train_df)
    X_test = X_test.drop(columns=categorical_features).join(encoded_test_df)

    return X_train,X_test


# app_name/utils.py
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def stepwise_selection(X, y, n_features_to_select=5, n_jobs=-1):
    model = LogisticRegression(solver='liblinear')
    rfe = RFE(model, n_features_to_select=n_features_to_select, step=1)
    fit = rfe.fit(X, y)
    selected_features = X.columns[fit.support_]
    return selected_features

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import base64

# Define the preprocessing function
def preprocess_data(X):
    """Preprocess the data by encoding categorical variables and scaling numerical ones."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, preprocessor



def train_logistic_regression(X_train, y_train):
    """
    Entraîne un modèle de régression logistique.
    
    Parameters:
    X_train (pd.DataFrame or np.array): Les caractéristiques de l'ensemble d'entraînement.
    y_train (pd.Series or np.array): La variable cible de l'ensemble d'entraînement.
    
    Returns:
    model (LogisticRegression): Le modèle de régression logistique entraîné.
    """
    # Initialiser et entraîner le modèle
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model

def summaryRL(reg,X_train):
    feature_name = X_train.columns.values
    # Stores the names of the columns of a dataframe in a variable.
    summary_table = pd.DataFrame(columns = ['Feature_name'], data = feature_name)
    # Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.
    summary_table['Coefficients'] = np.transpose(reg.coef_)
    # Creates a new column in the dataframe, called 'Coefficients',
    # with row values the transposed coefficients from the 'LogisticRegression' object.
    summary_table.index = summary_table.index + 1
    # Increases the index of every row of the dataframe with 1.
    summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
    # Assigns values of the row with index 0 of the dataframe.
    summary_table = summary_table.sort_index()
    
    return summary_table



class LogisticRegression_with_p_values:
    
    def __init__(self,*args,**kwargs):
        self.model = linear_model.LogisticRegression(*args,**kwargs)

    def fit(self,X,y):
        self.model.fit(X,y)
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X)
        Cramer_Rao = np.linalg.inv(F_ij)
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

def train_logistic_regreesion_pvalues(X_train,y_train):
    model = LogisticRegression_with_p_values()
    model.fit(X_train,y_train)
    return model

def decompose_variables(data):
    """
    Décompose les variables d'un DataFrame en variables catégorielles et numériques.

    Parameters:
    data (pd.DataFrame): Le DataFrame à traiter.

    Returns:
    tuple: (list, list) Une liste des noms des colonnes catégorielles et une liste des noms des colonnes numériques.
    """
    # Identifier les colonnes numériques
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Identifier les colonnes catégorielles
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return categorical_cols, numeric_cols

def optimize_hyperparameters(X_train, y_train):
    """
    Optimise les hyperparamètres du modèle de régression logistique.
    
    Parameters:
    X_train (pd.DataFrame or np.array): Les caractéristiques de l'ensemble d'entraînement.
    y_train (pd.Series or np.array): La variable cible de l'ensemble d'entraînement.
    
    Returns:
    best_model (LogisticRegression): Le meilleur modèle après optimisation des hyperparamètres.
    """
    param_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear']
    }
    
    log_reg = LogisticRegression(random_state=0, max_iter=10000)
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    return best_model

def modelwithoutbalance(data,target,features_selected):
    data=data[features_selected+[target]]
    categorical_features,numeric_cols=decompose_variables(data)
    X_train, X_test, y_train, y_test=split_data(data,target)
    X_train,X_test=encode_categorical_features(X_train,X_test,categorical_features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)
    X_train=pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test=pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    reg=train_logistic_regression(X_train, y_train)
    summary_table=summaryRL(reg,X_train)
    return summary_table,reg, X_train, X_test, y_train, y_test

def modelwithbalance(data,target,features_selected):
    data=data[features_selected+[target]]
    categorical_features,numeric_cols=decompose_variables(data)
    X_train, X_test, y_train, y_test=split_data(data,target)
    X_train,y_train=balance_train_data(X_train, y_train, categorical_features, random_state=None)
    X_train,X_test=encode_categorical_features(X_train,X_test,categorical_features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train=pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test=pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    reg=train_logistic_regression(X_train, y_train)
    summary_table=summaryRL(reg,X_train)
    return summary_table,reg, X_train, X_test, y_train, y_test
    



def print_performance_metrics(model, X, y, set_name):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    return report



def plot_confusion_matrix(model, X, y, set_name):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {set_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Correct path
    plots_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    filename = os.path.join(plots_dir, f'confusion_matrix_{set_name}.png')
    
    plt.savefig(filename)
    plt.close()
    
    # Return the relative path to the media directory
    return os.path.join(settings.MEDIA_URL, 'plots', f'confusion_matrix_{set_name}.png')

def remove_columns_with_missing_data(data, threshold):
    """
    Supprime les colonnes d'un DataFrame ayant un pourcentage de valeurs manquantes supérieur au seuil spécifié.

    Parameters:
    data (pd.DataFrame): Le DataFrame à traiter.
    threshold (float): Le seuil de pourcentage (entre 0 et 1) de valeurs manquantes à ne pas dépasser.

    Returns:
    tuple: (pd.DataFrame, list) Le DataFrame modifié sans les colonnes ayant trop de valeurs manquantes,
    et une liste des noms des colonnes supprimées.
    """
    # Calculer le pourcentage de valeurs manquantes pour chaque colonne
    missing_percent = data.isnull().mean()
    
    # Filtrer les colonnes où le pourcentage de valeurs manquantes est supérieur au seuil
    columns_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    
    # Supprimer les colonnes appropriées
    data_cleaned = data.drop(columns=columns_to_drop)
    
    return data_cleaned, columns_to_drop


def impute_missing_values(data):
    """
    Impute les valeurs manquantes dans un DataFrame.

    Parameters:
    data (pd.DataFrame): Le DataFrame à traiter.

    Returns:
    pd.DataFrame: Le DataFrame modifié avec les valeurs manquantes imputées.
    """
    # Imputer les valeurs manquantes pour les colonnes numériques avec la moyenne
    numeric_cols = data.select_dtypes(include='number').columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Imputer les valeurs manquantes pour les colonnes catégorielles avec le mode
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

    return data

def decompose_variables(data):
    """
    Décompose les variables d'un DataFrame en variables catégorielles et numériques.

    Parameters:
    data (pd.DataFrame): Le DataFrame à traiter.

    Returns:
    tuple: (list, list) Une liste des noms des colonnes catégorielles et une liste des noms des colonnes numériques.
    """
    # Identifier les colonnes numériques
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Identifier les colonnes catégorielles
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return categorical_cols, numeric_cols


import numpy as np
import pandas as pd
from scipy.stats import mode


# Function to detect missings using the IQR method
def detect_missing(x):
   return pd.isna(x)

# Function to replace missings using a specified method
import numpy as np
from scipy.stats import mode
import pandas as pd
import re

def convert_experience_to_numeric(value):
    """ Convertit les valeurs alphanumériques en années numériques. """
    if isinstance(value, str):
        if "year" in value:
            if "+" in value:  # Cas "10+ years"
                return int(re.search(r"\d+", value).group())  # Prend seulement le nombre
            elif "<" in value:  # Cas "< 1 year"
                return 0.5  # On peut considérer "< 1 year" comme 0.5 ans
            else:
                return int(re.search(r"\d+", value).group())  # Ex: "3 years" → 3
    return np.nan  # Si ce n'est pas une chaîne valide, retourne NaN

def replace_missings(x, method):
   
    """Replace missing values using the specified method."""
    x = np.array(x, dtype=object)
    missings = pd.isna(x)
    
    if method == "mode":
        if x.dtype.kind in "OSU":  # Gérer les colonnes de chaînes de caractères
            mode_value = pd.Series(x).mode().iloc[0] if not pd.Series(x).mode().empty else None
        else:
            mode_result = mode(x[~missings], nan_policy="omit")
            mode_value = mode_result.mode[0] if mode_result.mode.size > 0 else None
        
        if mode_value is not None:
            x[missings] = mode_value

    elif method == "mean":
        # Convertir en float et ignorer les NaN pour le calcul de la moyenne
        if np.issubdtype(x.dtype, np.number):
            mean_value = np.nanmean(x.astype(float))
            x[missings] = mean_value
        else:
            raise ValueError("La méthode 'mean' ne peut être appliquée qu'à des données numériques.")

    elif method == "median":
        # Convertir en float et ignorer les NaN pour le calcul de la médiane
        median_value = np.nanmedian(x)
        # Remplacer les valeurs manquantes par la médiane
        x[missings] = median_value


    return x




    # missings = detect_missing(x)
    
    
    # if method == "mode":
    #     mode_result = mode(x)
    #             # Vérifie si mode_result.mode est un tableau et contient des éléments
    #     if isinstance(mode_result.mode, np.ndarray) and len(mode_result.mode) > 0:
    #                 mode_value = mode_result.mode[0]
    #                 x[missings] = mode_value
    #     else:
    #         mode_value = mode_result.mode
    #     x[missings] = mode_value
    # elif method == "mean":
    #     mean_value =  np.nanmean(x)
    #     x[missings] = mean_value
    # elif method == "median":
    #     median_value = np.nanmedian(x)
    #     x[missings] = median_value
    
    # return x
# def treat_outliers(data, threshold=1.5, method='iqr', replace_with=None):
#     """
#     Traite les valeurs aberrantes dans un DataFrame.

#     Parameters:
#     data (pd.DataFrame): Le DataFrame à traiter.
#     threshold (float): Seuil pour identifier les valeurs aberrantes. Par défaut: 1.5.
#     method (str): Méthode pour détecter les valeurs aberrantes ('iqr' pour écart interquartile).
#                   Par défaut: 'iqr'.
#     replace_with (str or None): Optionnel. Si défini sur 'median' ou 'mean', remplace les valeurs aberrantes par
#                                 la médiane ou la moyenne. Par défaut: None (supprime les lignes).

#     Returns:
#     pd.DataFrame: Le DataFrame modifié avec les valeurs aberrantes traitées.
#     """
#     if method == 'iqr':
#         Q1 = data.quantile(0.25)
#         Q3 = data.quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - threshold * IQR
#         upper_bound = Q3 + threshold * IQR

#         # Identifier les lignes avec des valeurs aberrantes
#         outliers = ((data < lower_bound) | (data > upper_bound)).any(axis=1)

#         if replace_with == 'median':
#             data[outliers] = data.median()
#         elif replace_with == 'mean':
#             data[outliers] = data.mean()
#         else:
#             # Supprimer les lignes avec des valeurs aberrantes
#             data = data[~outliers]

#     return data


def remove_binned_columns(data):
    """
    Supprime les colonnes dont les noms commencent par 'binned_' dans un DataFrame pandas.
    
    Parameters:
    data (pd.DataFrame): Le DataFrame à traiter.
    
    Returns:
    pd.DataFrame: Le DataFrame sans les colonnes 'binned_'.
    """
    # Sélectionne les colonnes qui ne commencent pas par 'binned_'
    columns_to_keep = [col for col in data.columns if not col.startswith('binned_')]
    
    # Retourne un nouveau DataFrame avec uniquement les colonnes sélectionnées
    return data[columns_to_keep]



def load_model(path):
    return joblib.load(path)

class DataUploader(models.Model):
    # ...

    def change_column_type(self, column, new_type):
        # Récupérer les données
        data = self.data

        # Modifier le type de la colonne
        data[column] = data[column].astype(new_type)

        # Enregistrer les modifications
        self.data = data
        self.save()






    
