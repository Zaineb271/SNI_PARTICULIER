import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend sans interface graphique
import matplotlib.pyplot as plt
import seaborn as sns
import io 
import numpy as np 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import os


def data_overview(df):
    """
    Donne un aperçu général du jeu de données.
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.
    Returns:
    dict: Un dictionnaire contenant les informations générales du jeu de données.
    """
    overview = {}
    # Nombre de lignes et de colonnes
    overview['shape'] = df.shape
    # Types de données
    overview['dtypes'] = df.dtypes
    # Nombre de valeurs manquantes par colonne
    overview['missing_values'] = df.isnull().sum()/df.shape[0]*100
    # Statistiques descriptives pour les colonnes numériques
    overview['descriptive_stats'] = df.describe()
    # Aperçu des premières lignes du DataFrame
    overview['head'] = df.head()
    # Aperçu des dernières lignes du DataFrame
    overview['tail'] = df.tail()
    # Aperçu des valeurs uniques par colonne (optionnel)
    overview['unique_values'] = df.nunique()
    # Aperçu des informations générales du DataFrame
    buffer = io.StringIO()
    df.info(buf=buffer)
    overview['info'] = buffer.getvalue()
    return overview

def save_plot(filename):
    plt.savefig(filename)
    plt.close()

def plot_numeric_distribution(df, column, filename):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df[column], kde=False)
    plt.title('Histogramme')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[column])
    plt.title('Box Plot')
    
    plt.tight_layout()
    save_plot(filename)

def plot_categorical_distribution(df, column, filename):
    plt.figure(figsize=(10, 6))
    sns.countplot(y=df[column], order=df[column].value_counts().index)
    plt.title('Bar Plot')
    save_plot(filename)

def plot_missing_values(df, filename):
    if df.isnull().sum().sum() == 0:
        return False
    missing_values = df.isnull().sum()/df.shape[0]
    missing_values = missing_values[missing_values > 0]
    plt.figure(figsize=(10, 6))
    missing_values.plot(kind='bar')
    plt.title('Valeurs Manquantes par Colonne')
    plt.ylabel('Nombre de Valeurs Manquantes')
    save_plot(filename)
    return True
    

def plot_boxplot(df, feature, target, filename):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target, y=feature, data=df)
    plt.title(f'Box Plot de {feature} par {target}')
    save_plot(filename)

def plot_scatter_and_correlation(df, col1, col2, filename):
    correlation = df[col1].corr(df[col2])
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=col1, y=col2)
    plt.title(f'Scatter Plot de {col1} vs {col2}\nCorrélation de Pearson: {correlation:.2f}')
    save_plot(filename)

def plot_categorical_comparison(df, feature, target, filename):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')
    crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'Bar Plot Empilé de {feature} par {target}')
    plt.ylabel('Proportion')
    save_plot(filename)

def bivariate_analysis(df, target, plot_dir):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in numerical_cols:
        if col != target:
            scatter_filename = os.path.join(plot_dir, f'scatter_{col}_{target}.png')
            boxplot_filename = os.path.join(plot_dir, f'boxplot_{col}_{target}.png')
            plot_scatter_and_correlation(df, col, target, scatter_filename)
            plot_boxplot(df, col, target, boxplot_filename)

    for col in categorical_cols:
        if col != target:
            comparison_filename = os.path.join(plot_dir, f'categorical_comparison_{col}_{target}.png')
            plot_categorical_comparison(df, col, target, comparison_filename)

def plot_pie_chart(df, column, filename):
    data = df[column].value_counts()
    labels = data.index
    sizes = data.values
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(f'Diagramme Circulaire de {column}')
    plt.axis('equal')
    save_plot(filename)


def calculate_iv(df, feature, target):
    """
    Calculate the information value (IV) of a feature in a dataset.
    
    Parameters:
    df (pandas.DataFrame): the dataset
    feature (str): the name of the feature to calculate IV for
    target (str): the name of the target variable
    
    Returns:
    float: the information value (IV) of the feature
    """
    # Ensure the target column is numeric
    df[target] = pd.to_numeric(df[target], errors='coerce')
    
    # Drop rows with missing values
    df = df[[feature, target]].dropna()
    
    # Calculate the number of observations, and the number of good and bad cases
    n = df.shape[0]
    good = df[target].sum()  # Sum of '1's for the target (good cases)
    bad = n - good  # Total minus good gives bad cases
    
    # Get the unique values of the feature
    unique_values = df[feature].unique()
    
    iv = 0  # Initialize Information Value (IV)
    
    for value in unique_values:
        n1 = df[df[feature] == value].shape[0]  # Count of rows for this value
        good1 = df[(df[feature] == value) & (df[target] == 1)].shape[0]  # Good cases for this value
        bad1 = n1 - good1  # Bad cases for this value
        
        if good1 == 0 or bad1 == 0:  # Skip if there are no good or no bad cases
            continue
        
        # Calculate Weight of Evidence (WOE)
        woe = np.log((good1 / good) / (bad1 / bad))
        
        # Update Information Value (IV)
        iv += (good1 / good - bad1 / bad) * woe
    
    return iv

def categorize_iv(iv):
    """
    Categorize the information value (IV) of a feature based on commonly used cut-off values.
    
    Parameters:
    iv (float): the information value (IV) of a feature
    
    Returns:
    str: the category of the IV value (e.g., 'weak', 'moderate', 'strong')
    """
    if iv < 0.02:
        return 'not useful'
    elif iv < 0.1:
        return 'weak'
    elif iv < 0.3:
        return 'moderate'
    elif iv < 0.5:
        return 'strong'
    else:
        return 'suspicious'

def bin_variable(df, feature, bins=10, method='equal_width'):
    """
    Bin a continuous variable into discrete categories.
    
    Parameters:
    df (pandas.DataFrame): the dataset
    feature (str): the name of the feature to bin
    bins (int): the number of bins
    method (str): the binning method ('equal_width' or 'equal_frequency')
    
    Returns:
    pd.Series: the binned feature
    """
    if method == 'equal_width':
        return pd.cut(df[feature], bins=bins, duplicates='drop')
    elif method == 'equal_frequency':
        return pd.qcut(df[feature], q=bins, duplicates='drop')
    else:
        raise ValueError("Method must be 'equal_width' or 'equal_frequency'")

def calculate_bin_iv(df, binned_feature, target):
    """
    Calculate the information value (IV) of each bin in a binned feature.
    
    Parameters:
    df (pandas.DataFrame): the dataset
    binned_feature (str): the name of the binned feature
    target (str): the name of the target variable
    
    Returns:
    float: the information value (IV) of the binned feature
    """
    df = df[[binned_feature, target]].dropna()
    n = df.shape[0]
    good = df[target].sum()
    bad = n - good
    iv = 0
    for bin in df[binned_feature].unique():
        n1 = df[df[binned_feature] == bin].shape[0]
        good1 = df[(df[binned_feature] == bin) & (df[target] == 1)].shape[0]
        bad1 = n1 - good1
        if good1 == 0 or bad1 == 0:
            continue
        woe = np.log((good1 / good) / (bad1 / bad))
        iv += (good1 / good - bad1 / bad) * woe
    return iv

def merge_bins(df, binned_feature, target):
    """
    Merge adjacent bins with similar IV values.
    
    Parameters:
    df (pandas.DataFrame): the dataset
    binned_feature (str): the name of the binned feature
    target (str): the name of the target variable
    
    Returns:
    pd.Series: the merged binned feature
    """
    while True:
        bins = df[binned_feature].unique()
        bin_pairs = [(bins[i], bins[i+1]) for i in range(len(bins) - 1)]
        min_diff = float('inf')
        merge_pair = None
        for bin1, bin2 in bin_pairs:
            df['temp'] = df[binned_feature].replace({bin1: f'{bin1},{bin2}', bin2: f'{bin1},{bin2}'})
            iv = calculate_bin_iv(df, 'temp', target)
            diff = abs(iv - calculate_bin_iv(df, binned_feature, target))
            if diff < min_diff:
                min_diff = diff
                merge_pair = (bin1, bin2)
        if min_diff > 0.1:  # A threshold to stop merging
            break
        df[binned_feature] = df[binned_feature].replace({merge_pair[0]: f'{merge_pair[0]},{merge_pair[1]}', merge_pair[1]: f'{merge_pair[0]},{merge_pair[1]}'})
    return df[binned_feature]

def calculate_iv_table_with_binning(df, target, bins=10, method='equal_width'):
    """
    Calculate the IV for all features in the dataset with binning and categorize them.
    
    Parameters:
    df (pandas.DataFrame): the dataset
    target (str): the name of the target variable
    bins (int): the number of bins for continuous features
    method (str): the binning method ('equal_width' or 'equal_frequency')
    
    Returns:
    pd.DataFrame: a DataFrame with columns 'Feature', 'IV', 'Category', and 'Binned_Feature'
    """
    iv_list = []
    for feature in df.columns:
        if feature == target:
            continue
        if df[feature].dtype in ['float64', 'int64']:
            binned_feature = bin_variable(df, feature, bins, method)
            df[f'binned_{feature}'] = binned_feature
            #df[f'binned_{feature}'] = merge_bins(df, f'binned_{feature}', target)
            iv = calculate_bin_iv(df, f'binned_{feature}', target)
            binned_feature_str = f'binned_{feature}'
        else:
            iv = calculate_iv(df, feature, target)
            binned_feature_str = None
        category = categorize_iv(iv)
        iv_list.append((feature, iv, category, binned_feature_str))
    
    iv_df = pd.DataFrame(iv_list, columns=['Feature', 'IV', 'Category', 'Binned_Feature'])
    return iv_df



def chi2_test(df, target):
    from scipy.stats import chi2_contingency
    """
    Effectue un test du chi-carré pour toutes les features qualitatives du DataFrame par rapport à la target.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.
    target (str): Le nom de la colonne cible.
    
    Returns:
    pd.DataFrame: Un DataFrame avec les features, les p-values et une indication d'indépendance.
    """
    results = []
    
    for feature in df.columns:
        if feature == target or df[feature].dtype not in ['object', 'category']:
            continue
        
        # Construire une table de contingence
        contingency_table = pd.crosstab(df[feature], df[target])
        
        # Effectuer le test du chi-carré
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)
        
        # Ajouter les résultats à la liste
        results.append((feature, chi2, p_val, 'Independent' if p_val > 0.05 else 'Dependent'))
    
    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results, columns=['Feature', 'Chi2_value', 'P_value', 'Independence'])
    return results_df



def correlation_matrix(df, plot=False):
    """
    Calcule et retourne la matrice de corrélation pour les features numériques d'un DataFrame.
    
    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.
    plot (bool): Si True, affiche une heatmap de la matrice de corrélation. Par défaut: False.
    
    Returns:
    pd.DataFrame: La matrice de corrélation.
    """
    # Sélectionner uniquement les colonnes numériques
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculer la matrice de corrélation
    corr_matrix = numeric_df.corr()
    
    if plot:
        # Afficher la heatmap de la matrice de corrélation
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
        plt.title('Matrice de Corrélation')
        save_plot('media/plots/correlation_matrix.png')
        plt.show()
    
    return corr_matrix



def feature_selection(X_train, y_train, n_features_to_select):
    """
    Sélectionne les meilleures caractéristiques avec RFE.
    
    Parameters:
    X_train (pd.DataFrame): Les caractéristiques de l'ensemble d'entraînement.
    y_train (pd.Series): La variable cible de l'ensemble d'entraînement.
    n_features_to_select (int): Le nombre de caractéristiques à sélectionner.
    
    Returns:
    X_train_selected (pd.DataFrame): Les caractéristiques sélectionnées.
    selected_features (list): Liste des noms des caractéristiques sélectionnées.
    """
    # Initialize Linear Regression model for regression task
    reg_model = LinearRegression()
    
    # Initialize RFE with the model
    rfe = RFE(reg_model, n_features_to_select=n_features_to_select)
    
    # Fit RFE to the data
    rfe.fit(X_train, y_train)
    
    # Select features based on RFE
    X_train_selected = X_train.loc[:, rfe.support_]  # Use the support array to select columns
    
    # Get the selected feature names
    selected_features = X_train.columns[rfe.support_].tolist()  # Convert to a list
    
    return X_train_selected, selected_features  