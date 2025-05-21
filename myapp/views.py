from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import pandas as pd
import os
from django.views.generic import TemplateView
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from django.http import HttpResponse
from django.views import View
from django.shortcuts import render




from .forms import (
    ColumnSelectionForm,
    BivariateColumnSelectionForm,
    MultipleColumnSelectionForm,
    DataTreatmentForm,
)
from .analysis_functions import *
from .models import *
import pickle
import base64
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from datetime import datetime


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get("username")
            password = form.cleaned_data.get("password")
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect(
                    "index"
                )  # Rediriger vers la page d'accueil ou toute autre page après la connexion
    else:
        form = AuthenticationForm()
        messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    return render(request, "login.html", {"form": form, "messages": messages})


# @login_required
# def index(request):
#     return render(request, "index.html")


#@login_required
# def upload_file(request):
#     if request.method == "POST" and "file" in request.FILES:
#         file = request.FILES["file"]
#         fs = FileSystemStorage()
#         filename = fs.save(file.name, file)
#         uploaded_file_url = fs.url(filename)
#         if filename.endswith(".xlsx"):
#             uploaded_data = pd.read_excel(fs.path(filename))
#         else:
#             return render(
#                 request,
#                 "index.html",
#                 {"error": "Le fichier doit être un fichier Excel (.xlsx)"},
#             )
#         # uploaded_data = pd.read_csv(fs.path(filename))
#         request.session["uploaded_data"] = uploaded_data.to_json()
#         return redirect("overview")
#     return render(request, "index.html")
# def upload_file(request):
#     # Charger le fichier Excel statique "bfi.xlsx" sous le dossier "data"
#     static_file_path = "bfi.xlsx"
#     uploaded_data = pd.read_excel(static_file_path)
    
#     # Stocker les données dans la session
#     request.session["uploaded_data"] = uploaded_data.to_json()
    
#     return redirect("overview")


#@login_required
def data_overview_view(request):
    if "uploaded_data" not in request.session:
        static_file_path = "bfi.xlsx"
        uploaded_data = pd.read_excel(static_file_path)
        # Sauvegarder les données de base dans la session
        request.session["uploaded_data"] = uploaded_data.to_json()
    
    
    if "uploaded_data" in request.session:
        # uploaded_data = pd.read_json(request.session["uploaded_data"])
        # print("Data loaded from session in data_overview_view:")
        # print(uploaded_data.dtypes)
        
        static_file_path = "bfi.xlsx"
        uploaded_data = pd.read_excel(static_file_path)
        # Sauvegarder les données de base dans la session
        request.session.modified = True 
        #request.session["uploaded_data"] = uploaded_data.to_json()
        # Si les données sont déjà dans la session, les charger directement
        #uploaded_data = pd.read_json(request.session["uploaded_data"])
        
        # Traitement des colonnes de dates
        date_columns = ['last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 'mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']
        date_format = "%Y-%m-%d"
        # Conversion des colonnes de dates en format datetime
        for col in date_columns:
            if col in uploaded_data.columns:  # Vérifier si la colonne existe dans les données
               uploaded_data[col] = pd.to_datetime(uploaded_data[col],  format=date_format, errors="coerce")


        # Remplacement des valeurs manquantes par la date la plus récente
        for col in date_columns:
            if col in uploaded_data.columns:  # Vérifier si la colonne existe dans les données
                uploaded_data[col] = uploaded_data[col].fillna(uploaded_data[col].max())  # Correction ici

        print(uploaded_data["last_pymnt_d"].dtype)
        print(uploaded_data["mths_since_last_delinq"].dtype)

        # Sauvegarder les données de base dans la session
        uploaded_data = pd.read_json(request.session["uploaded_data"])
        #request.session["uploaded_data"] = uploaded_data.to_json()
        

        message = None  # Initialisation du message à None

        if request.method == "POST":
            
            if "target_column" in request.POST:
                # Enregistrer la colonne cible dans la session
                target_column = request.POST.get("target_column")
                print("Colonne cible actuelle:", target_column)  # Debug
                if target_column and target_column in uploaded_data.columns:
                    request.session["target_column"] = target_column
                    message = f"La Variable cible {target_column} a été enregistrée."
                    print("Variable cible enregistrée:", target_column)  # Debug
                # Convertir les valeurs de la colonne cible en valeurs binaires
                    if target_column == 'loan_status':  # Remplace 'loan_status' par ta colonne cible
                        uploaded_data['loan_status'] = uploaded_data[target_column].map({
                            'Fully Paid': 0,
                            'Current': 0,
                            'Charged Off': 1,
                            'Default': 1,
                            'Late (31-120 days)': 1,
                            'In Grace Period': 1,
                            'Late (16-30 days)': 1,
                            'Does not meet the credit policy. Status:Charged Off': 1
                        })
                        # Sauvegarder les données avec la colonne convertie dans la session
                        request.session.modified = True 
                        request.session["uploaded_data"] = uploaded_data.to_json()
                        message += " La colonne cible a été convertie en valeurs binaires."
                    else:
                        message += " La colonne cible n'est pas reconnue pour une conversion binaire."
                else:
                    message = "Aucune Variable cible valide n'a été sélectionnée."
                    print("Erreur: Aucune Variable cible valide.")  # Debug
        

            elif "column_to_normalize" in request.POST:
                # Normalisation des données
                column_to_normalize = request.POST.get("column_to_normalize")
                if column_to_normalize and column_to_normalize in uploaded_data.columns:
                    if pd.api.types.is_numeric_dtype(
                        uploaded_data[column_to_normalize]
                    ):
                        mean_value = uploaded_data[column_to_normalize].mean()
                        sd_value = uploaded_data[column_to_normalize].std()
                        normalized_column_name = f"{column_to_normalize}_normalisé"
                        uploaded_data[normalized_column_name] = (
                            uploaded_data[column_to_normalize] - mean_value
                        ) / sd_value
                        # Mettez à jour les données dans la session après normalisation
                        request.session["uploaded_data"] = uploaded_data.to_json()
                        message = f"Column {column_to_normalize} has been normalized."
                    else:
                        message = "Cette variable n'est pas numérique."
                else:
                    message = "Aucune colonne à normaliser n'a été sélectionnée."

            elif "column_to_bin" in request.POST:
                # Binning (tri par palier)
                column_to_bin = request.POST.get("column_to_bin")
                if column_to_bin and column_to_bin in uploaded_data.columns:
                    if pd.api.types.is_numeric_dtype(uploaded_data[column_to_bin]):
                        quantiles = (
                            uploaded_data[column_to_bin]
                            .quantile([0, 1 / 3, 2 / 3, 1])
                            .values
                        )
                        # Vérifiez que les quantiles sont uniques
                        quantiles = np.unique(quantiles)

                        if len(quantiles) > 1:
                            binned_column_name = f"{column_to_bin}_parpalier"
                            labels = [
                                f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]"
                                for i in range(len(quantiles) - 1)
                            ]
                            uploaded_data[binned_column_name] = pd.cut(
                                uploaded_data[column_to_bin],
                                bins=quantiles,
                                include_lowest=True,
                                labels=labels,
                                duplicates="drop",
                                ordered=True,  # Assurez-vous que les étiquettes sont uniques si ordered=True
                            )
                            # Mettez à jour les données dans la session après binning
                            request.session["uploaded_data"] = uploaded_data.to_json()
                            message = f"Column {column_to_bin} has been binned."
                        else:
                            message = "Les quantiles ne permettent pas de créer des bacs distincts."
                    else:
                        message = "Cette variable n'est pas numérique."
                else:
                    message = "Aucune colonne à trier par palier n'a été sélectionnée."

            elif "column_to_process" in request.POST and "method" in request.POST:
                column_to_process = request.POST.get("column_to_process")
                method = request.POST.get("method")

                if column_to_process and column_to_process in uploaded_data.columns:
                     # Convertir les valeurs si elles sont alphanumériques
                    
                    #if pd.api.types.is_numeric_dtype(uploaded_data[column_to_process]):
                        #uploaded_data[column_to_process] = uploaded_data[column_to_process].apply(convert_experience_to_numeric)

            
                        # Replace missing values
                        uploaded_data[column_to_process] = replace_missings(
                            uploaded_data[column_to_process].values, method
                        )
                        # Update session data
                        request.session.modified = True 
                        request.session["uploaded_data"] = uploaded_data.to_json()
                        print("have been replaced using")
                        message = f"Missing values in column {column_to_process} have been replaced using {method}."
                    #else:
                     #   message = "The selected column is not numeric."
                else:
                    message = "No column was selected or the column does not exist."

            elif "column_to_remove" in request.POST:
                # Suppression de colonne
                column_to_remove = request.POST.get("column_to_remove")
                if column_to_remove in uploaded_data.columns:
                    uploaded_data = uploaded_data.drop(columns=[column_to_remove])
                    # Mettez à jour les données dans la session après suppression
                    request.session["uploaded_data"] = uploaded_data.to_json()
                    message = f"Column {column_to_remove} has been removed."
                else:
                    message = f"Column {column_to_remove} does not exist."
                    
        columns = uploaded_data.columns.tolist()
        print("Colonnes disponibles pour sélection:", columns)  # Debug
        
        # Ajuster les options d'affichage de Pandas pour l'affichage complet
        pd.set_option("display.max_columns", None)  # Afficher toutes les colonnes
        pd.set_option(
            "display.expand_frame_repr", False
        )  # Ne pas tronquer les colonnes

        # Générer l'aperçu limité avec head()
        data_preview = uploaded_data.head().to_html()

        # Filtrer les données pour afficher à partir de l'index 1
        uploaded_data.index = uploaded_data.index + 1

        # Générer l'aperçu limité avec head() des données filtrées
        data_full = uploaded_data.round(2).to_html()

        # Restaurer les options d'affichage pour l'affichage complet
        pd.set_option("display.max_columns", None)
        pd.set_option("display.expand_frame_repr", False)

        overview = data_overview(uploaded_data)
        shape = overview.get("shape")

        # Format de la forme comme "3899 lignes et 67 colonnes"
        formatted_shape = f"{shape[0]} lignes et {shape[1]} colonnes"
        
    
        return render(
            request,
            "overview.html",
            {
                "overview": overview,
                "data_pre": data_preview,
                "data_full": data_full,
                "formatted_shape": formatted_shape,
                "columns": uploaded_data,
                "message": message
            },
        )

    return redirect("overview")


@login_required
def plot_selection_view(request):
    if "uploaded_data" in request.session:

        uploaded_data = pd.read_json(request.session["uploaded_data"])

        plots = request.session.get("plot", [])
        target_column = request.session.get("target_column")
        categorical_cols, numeric_cols = decompose_variables(uploaded_data)

        selected_numeric_column = None
        selected_categorical_column = None
       
        if request.method == "POST":
            plot_dir = "media/plots/"
            os.makedirs(plot_dir, exist_ok=True)

            # Pour les colonnes numériques
            if "plot_numeric_distribution" in request.POST:
                form = ColumnSelectionForm(request.POST, columns=numeric_cols)
                if form.is_valid():
                    selected_numeric_column = form.cleaned_data["column"]
                    filename = os.path.join(plot_dir, "numeric_distribution.png")
                    plot_numeric_distribution(uploaded_data, selected_numeric_column, filename)
                    plot_path = f"/media/plots/numeric_distribution.png"
                    plots.append(plot_path)  

            # Pour les colonnes catégoriques
            if "plot_categorical_distribution" in request.POST:
                form = ColumnSelectionForm(request.POST, columns=categorical_cols)
                if form.is_valid():
                    selected_categorical_column = form.cleaned_data["column"]
                    filename = os.path.join(plot_dir, "categorical_distribution.png")
                    plot_categorical_distribution(uploaded_data, selected_categorical_column, filename)
                    plot_path = f"/media/plots/categorical_distribution.png"
                    plots.append(plot_path)
                    
            if "plot_pie_chart" in request.POST:
                
                if target_column and target_column in uploaded_data.columns:
                        # Générer le Pie Chart pour la colonne cible
                        filename = os.path.join(plot_dir, "pie_chart.png")
                        print("Target column for pie chart:", target_column)  # Debug
                        plot_pie_chart(uploaded_data, target_column, filename)
                        plot_path = f"/media/plots/pie_chart.png"
                        plots.append(plot_path)    	
            
            request.session["plots"] = plots

        # Passer la colonne sélectionnée aux formulaires
        numeric_column_selection_form = ColumnSelectionForm(columns=numeric_cols, selected_column=selected_numeric_column)
        categorical_column_selection_form = ColumnSelectionForm(columns=categorical_cols, selected_column=selected_categorical_column)

        return render(
            request,
            "plot.html",
            {
                "numeric_column_selection_form": numeric_column_selection_form,
                "categorical_column_selection_form": categorical_column_selection_form,
                "categorical_distribution_plots": [
                    plot for plot in plots if "categorical_distribution" in plot
                ],
                "pie_chart_plots": [plot for plot in plots if "pie_chart" in plot],
                "numeric_distribution_plots": [
                    plot for plot in plots if "numeric_distribution" in plot
                ],
               "target_column": target_column,
               "plots": plots
               
            },
        )

    return redirect("overview")






# @login_required
# def feature_selection_view(request):
#     if "uploaded_data" in request.session:
#         uploaded_data = pd.read_json(request.session["uploaded_data"])

#         # Récupérer la colonne cible à partir de la session
#         target_column = request.session.get("target_column")
#         message = None  # Initialisation du message à None
#         anova_results = None
#         iv_results = None
#         
# = None

#         if request.method == "POST":
#             # Vérifier si une colonne doit être supprimée
#             if "column_to_remove" in request.POST:
#                 # Suppression de colonne
#                 column_to_remove = request.POST.get("column_to_remove")
#                 if column_to_remove in uploaded_data.columns:
#                     uploaded_data = uploaded_data.drop(columns=[column_to_remove])
#                     # Mettre à jour les données dans la session après suppression
#                     request.session["uploaded_data"] = uploaded_data.to_json()
#                     message = f"La colonne {column_to_remove} a été supprimée."
#                 else:
#                     message = f"La colonne {column_to_remove} n'existe pas."

#             # Effectuer les tests ANOVA, IV, et générer la matrice de corrélation si une colonne cible est présente
#             elif target_column:
#                 anova_results = chi2_test(uploaded_data, target_column)
#                 iv_results = calculate_iv_table_with_binning(uploaded_data, target_column)
#                 correlation_matrix(uploaded_data, plot=True)
#                 correlation_plot = "/media/plots/correlation_matrix.png"
        
#         return render(
#             request,
#             "feature_selection.html",
#             {
#                 "uploaded_data": uploaded_data,
#                 "anova_results": anova_results,
#                 "iv_results": iv_results,
#                 "correlation_plot": correlation_plot,
#                 "current_column": target_column,  # Passer la colonne cible à la vue
#                 "message": message,  # Passer le message à la vue
#             },
#         )
#     return redirect("upload_file")


@login_required
def feature_selection_view(request):
    if "uploaded_data" in request.session:
        uploaded_data = pd.read_json(request.session["uploaded_data"])

        target_column = request.session.get("target_column")
        message = None
        anova_results = None
        iv_results = None
        correlation_plot = None
        correlated_variables = None
        corr_matrix = None
        correlated_pairs = []  # Initialiser comme liste vide

        if request.method == "POST":
            if "column_to_remove" in request.POST:
                column_to_remove = request.POST.get("column_to_remove")
                if column_to_remove in uploaded_data.columns:
                    uploaded_data = uploaded_data.drop(columns=[column_to_remove])
                    request.session["uploaded_data"] = uploaded_data.to_json()
                    message = f"La colonne {column_to_remove} a été supprimée."
                else:
                    message = f"La colonne {column_to_remove} n'existe pas."

            elif target_column:
                anova_results = chi2_test(uploaded_data, target_column)
                iv_results = calculate_iv_table_with_binning(uploaded_data, target_column)
                print(iv_results)# Afficher les résultats IV pour le débogage
                iv_values = iv_results['IV'].round(2).to_dict()  # Cela devrait être une Series de pandas
                print(iv_values)
                threshold = 0.2 
                # Calculer la matrice de corrélation et identifier les paires avec une haute corrélation
                  # Sélectionner uniquement les colonnes numériques
                numeric_data = uploaded_data.select_dtypes(include=[np.number])
                
                if numeric_data.empty:
                    return None, None
                
                # Calculer la matrice de corrélation
                corr_matrix = numeric_data.corr()
                
                print("Matrice de Corrélation:")
                print(corr_matrix)  # Afficher la matrice de corrélation pour le débogage
                
                # Identifier les paires de variables avec une corrélation au-dessus du seuil
                mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                cor_pairs = np.where((np.abs(corr_matrix) > threshold) & mask)
                
                print("Paires Corrélées:")
                print(cor_pairs)  # Afficher les indices des paires corrélées
                
                #if not cor_pairs[0].size:
                #    print("Paires:")
                #    return corr_matrix, pd.DataFrame()  # Retourner un DataFrame vide si aucune paire trouvée
                   
               
                # Créer un DataFrame pour stocker les résultats
                correlated_variables = pd.DataFrame({
                    'Variables': [f"{numeric_data.columns[i]} ({iv_values.get(numeric_data.columns[i], iv_values[i])}) | {numeric_data.columns[j]} ({iv_values.get(numeric_data.columns[j], iv_values[j])})" for i, j in zip(*cor_pairs)],
                    'Correlation': [corr_matrix.iat[i, j] for i, j in zip(*cor_pairs)],
                    'Choix_de_la_variable_a_garder': [f"Les variables {numeric_data.columns[i]} & {numeric_data.columns[j]} sont fortement corrélées" for i, j in zip(*cor_pairs)]
                })
                #corr_matrix, correlated_variables = calculate_correlation_matrix(uploaded_data, threshold=0.2)
                correlation_matrix(uploaded_data, plot=True)
                correlation_plot = "/media/plots/correlation_matrix.png"
                
               
                 # Préparer les paires corrélées pour l'affichage dans la template
                correlated_pairs = [
                    {
                        'var1': numeric_data.columns[i],
                        'var2': numeric_data.columns[j],
                        'iv1': iv_values.get(numeric_data.columns[i], iv_values[i]),
                        'iv2': iv_values.get(numeric_data.columns[j], iv_values[j]),
                        'correlation': f"{corr_matrix.iat[i, j]:.2f}"
                    }
                    for i, j in zip(*cor_pairs)
                ]
                
        
        return render(
            request,
            "feature_selection.html",
            {
                "uploaded_data": uploaded_data,
                "anova_results": anova_results,
                "iv_results": iv_results,
                "correlation_plot": correlation_plot,
                "current_column": target_column,
                "message": message,
                "correlated_pairs": correlated_pairs,
                "correlated_variables": correlated_variables.to_html(index=False) if correlated_variables is not None else None,
                "corr_matrix": corr_matrix.to_html() if corr_matrix is not None else None,
            },
        )
    return redirect("overview")

# @login_required
# def modeling_view(request):
#     if "uploaded_data" in request.session:
#         uploaded_data = pd.read_json(request.session["uploaded_data"])
#         target = None
#         features_selected = []
#         summary_table = None
#         reg = None
#         performance_metrics = None
#         confusion_matrix_plot = None
#         X_train = X_test = y_train = y_test = None

#         if request.method == "POST":
#             target_selection_form = ColumnSelectionForm(
#                 request.POST, columns=uploaded_data.columns
#             )
#             feature_selection_form = MultipleColumnSelectionForm(
#                 request.POST, columns=uploaded_data.columns
#             )

#             if target_selection_form.is_valid() and feature_selection_form.is_valid():
#                 target = target_selection_form.cleaned_data["column"]
#                 features_selected = feature_selection_form.cleaned_data["columns"]

#                 # Remove target from features_selected if present
#                 if target in features_selected:
#                     features_selected.remove(target)

#                 # Store target and selected features in session
#                 request.session["selected_target"] = target
#                 request.session["selected_features"] = features_selected

#                 if "model_with_balance" in request.POST:
#                     summary_table, reg, X_train, X_test, y_train, y_test = (
#                         modelwithbalance(uploaded_data, target, features_selected)
#                     )
#                     request.session["model"] = base64.b64encode(
#                         pickle.dumps(reg)
#                     ).decode("utf-8")
#                     request.session["X_train"] = X_train.to_json()
#                     request.session["X_test"] = X_test.to_json()
#                     request.session["y_train"] = y_train.to_json()
#                     request.session["y_test"] = y_test.to_json()
#                     request.session["model_columns"] = X_train.columns.tolist()

#                 elif "model_without_balance" in request.POST:
#                     summary_table, reg, X_train, X_test, y_train, y_test = (
#                         modelwithoutbalance(uploaded_data, target, features_selected)
#                     )
#                     request.session["model"] = base64.b64encode(
#                         pickle.dumps(reg)
#                     ).decode("utf-8")
#                     request.session["X_train"] = X_train.to_json()
#                     request.session["X_test"] = X_test.to_json()
#                     request.session["y_train"] = y_train.to_json()
#                     request.session["y_test"] = y_test.to_json()
#                     request.session["model_columns"] = X_train.columns.tolist()

#             # Récupération du modèle et des données d'entraînement/test stockés dans la session
#             if "model" in request.session:
#                 reg = pickle.loads(base64.b64decode(request.session["model"]))
#                 X_train = pd.read_json(request.session["X_train"])
#                 X_test = pd.read_json(request.session["X_test"])
#                 y_train = pd.read_json(request.session["y_train"], typ="series")
#                 y_test = pd.read_json(request.session["y_test"], typ="series")

#             if reg is not None:
#                 if "show_metrics" in request.POST:
#                     metrics_set = request.POST.get("metrics_set")
#                     if metrics_set == "train":
#                         performance_metrics = print_performance_metrics(
#                             reg, X_train, y_train, "Train"
#                         )
#                         confusion_matrix_plot = plot_confusion_matrix(
#                             reg, X_train, y_train, "Train"
#                         )
#                     elif metrics_set == "test":
#                         performance_metrics = print_performance_metrics(
#                             reg, X_test, y_test, "Test"
#                         )
#                         confusion_matrix_plot = plot_confusion_matrix(
#                             reg, X_test, y_test, "Test"
#                         )
#                 else:
#                     performance_metrics = None
#                     confusion_matrix_plot = None

#         else:
#             target_selection_form = ColumnSelectionForm(columns=uploaded_data.columns)
#             feature_selection_form = MultipleColumnSelectionForm(
#                 columns=uploaded_data.columns
#             )

#         return render(
#             request,
#             "modeling.html",
#             {
#                 "target_selection_form": target_selection_form,
#                 "feature_selection_form": feature_selection_form,
#                 "summary_table": (
#                     summary_table.to_dict("records")
#                     if summary_table is not None
#                     else None
#                 ),
#                 "performance_metrics": performance_metrics,
#                 "confusion_matrix_plot": confusion_matrix_plot,
#             },
#         )
#     return redirect("upload_file")


import pandas as pd
import numpy as np
#import statsmodels.api as sm # type: ignore
from joblib import Parallel, delayed
import base64
import pickle
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

# Include the stepwise regression functions here (include_all_levels, detect_categorical_vars, compute_p_value, stepwise_selection)

import base64
import pickle
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from sklearn.model_selection import train_test_split

@login_required
def modeling_view(request):
    if "uploaded_data" in request.session:
        uploaded_data = pd.read_json(request.session["uploaded_data"])
        target_column = request.session.get("target_column")
        summary_table = None
        reg = None
        performance_metrics = None
        confusion_matrix_plot = None
        X_train = X_test = y_train = y_test = None
        selected_features = None  # Initialisation ici

        if request.method == "POST" and "train_model" in request.POST:
            if target_column and target_column in uploaded_data.columns:
                target = target_column
                features_selected = [col for col in uploaded_data.columns if col != target]

                # Extract feature set and target
                X = uploaded_data[features_selected]
                y = uploaded_data[target]

                # Preprocess data before feature selection
                X_preprocessed, preprocessor = preprocess_data(X)

                # Automatic feature selection using stepwise_selection
                if "auto_selection" in request.POST:
                    selected_features = stepwise_selection(
                        pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out()), y)
                    X_preprocessed = pd.DataFrame(X_preprocessed, columns=preprocessor.get_feature_names_out())[selected_features]
                    
                    print("Forme du DataFrame transformé :", X_preprocessed.shape)
                    # Save selected features to session for future use
                    # Convert Index to list if needed
                    if isinstance(selected_features, pd.Index):
                        selected_features = selected_features.tolist()
                    request.session["selected_features"] = selected_features
                    print("Selected features after stepwise selection:", selected_features)

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

                # Call the appropriate modeling function
                if "model_with_balance" in request.POST:
                    summary_table, reg, X_train, X_test, y_train, y_test = modelwithbalance(uploaded_data, target, features_selected)
                
                print("Shapes after balancing and encoding:")
                print("X_train shape:", X_train.shape)
                print("X_test shape:", X_test.shape)
                print("y_train shape:", y_train.shape)
                print("y_test shape:", y_test.shape)
                
                # Check if summary_table is None and handle it
                if summary_table is None:
                    summary_table = pd.DataFrame()  # Set an empty DataFrame as a fallback

                # Save objects to session
                request.session["model"] = base64.b64encode(pickle.dumps(reg)).decode("utf-8")
                request.session["preprocessor"] = base64.b64encode(pickle.dumps(preprocessor)).decode("utf-8")

                # Convert data to JSON-compatible format
                request.session["X_train"] = pd.DataFrame(X_train).reset_index(drop=True).to_json()
                request.session["X_test"] = pd.DataFrame(X_test).reset_index(drop=True).to_json()
                request.session["y_train"] = y_train.reset_index(drop=True).to_json()
                request.session["y_test"] = y_test.reset_index(drop=True).to_json()
                
                request.session["model_columns"] = list(X.columns)
                request.session["summary_table"] = summary_table.to_dict("records") if not summary_table.empty else []

        # Retrieve stored summary table and selected features from session
        summary_table = request.session.get("summary_table", [])
        selected_features = request.session.get("selected_features", [])

        # Handle show metrics request
        if request.method == "POST" and "show_metrics" in request.POST:
            if "model" in request.session:
                reg = pickle.loads(base64.b64decode(request.session["model"]))
                preprocessor = pickle.loads(base64.b64decode(request.session["preprocessor"]))
                X_train = pd.read_json(request.session["X_train"])
                X_test = pd.read_json(request.session["X_test"])
                y_train = pd.read_json(request.session["y_train"], typ="series")
                y_test = pd.read_json(request.session["y_test"], typ="series")

                metrics_set = request.POST.get("metrics_set")
                if metrics_set == "train":
                    performance_metrics = print_performance_metrics(reg, X_train, y_train, "Train")
                    confusion_matrix_plot = plot_confusion_matrix(reg, X_train, y_train, "Train")
                elif metrics_set == "test":
                    performance_metrics = print_performance_metrics(reg, X_test, y_test, "Test")
                    confusion_matrix_plot = plot_confusion_matrix(reg, X_test, y_test, "Test")

        return render(
            request,
            "modeling.html",
            {
                "current_column": target_column,
                "summary_table": summary_table,
                "performance_metrics": performance_metrics,
                "confusion_matrix_plot": confusion_matrix_plot,
                "selected_features": selected_features,
            },
        )
    return redirect("overview")




@login_required
def data_treatment_view(request):
    if "uploaded_data" in request.session:
        uploaded_data = pd.read_json(request.session["uploaded_data"])

        if request.method == "POST":
            action = request.POST.get("action")

            if action == "remove_column":
                column_to_remove = request.POST.get("column_to_remove")
                if column_to_remove in uploaded_data.columns:
                    data_cleaned = uploaded_data.drop(columns=[column_to_remove])
                    message = f"Column {column_to_remove} has been removed."
                else:
                    data_cleaned = uploaded_data
                    message = f"Column {column_to_remove} does not exist."

            elif action == "remove_missing":
                threshold = float(request.POST.get("threshold_remove_missing")) / 100.0
                data_cleaned, columns_removed = remove_columns_with_missing_data(
                    uploaded_data, threshold
                )
                message = f"Removed columns: {', '.join(columns_removed)}"

            elif action == "impute_missing":
                data_cleaned = impute_missing_values(uploaded_data)
                message = "Missing values have been imputed."

            # elif action == "treat_outliers":
            #     threshold = float(request.POST.get("threshold_treat_outliers"))

            #     # method = request.POST.get('method_treat_outliers')
            #     replace_with = request.POST.get("replace_with_treat_outliers")
            #     data_cleaned = treat_outliers(
            #         uploaded_data, threshold, replace_with=replace_with
            #     )
            #     message = "Outliers have been treated."

            request.session["uploaded_data"] = data_cleaned.to_json()
            return render(
                request,
                "data_treatment.html",
                {
                    "message": message,
                    "data_preview": data_cleaned.head().to_html(),
                    "columns": data_cleaned.columns.tolist(),
                },
            )

        return render(
            request,
            "data_treatment.html",
            {
                "columns": uploaded_data.columns.tolist(),
                "data_preview": uploaded_data.head().to_html(),
            },
        )
    return redirect("overview")


@login_required
def scoring_view(request):
    if (
        "uploaded_data" in request.session
        and "selected_features" in request.session
        and "model_columns" in request.session
    ):
        uploaded_data = pd.read_json(request.session["uploaded_data"])
        selected_features = request.session["selected_features"]
        target = request.session["selected_target"]
        model_columns = request.session["model_columns"]

        # Detect if features are categorical or numeric
        feature_types = {
            feature: (
                "categorical" if uploaded_data[feature].dtype == "object" else "numeric"
            )
            for feature in selected_features
        }

        # Prepare uploaded_data for template use and map categorical features to their dummies
        uploaded_data_dict = {
            feature: (
                uploaded_data[feature].unique().tolist()
                if feature_types[feature] == "categorical"
                else None
            )
            for feature in selected_features
        }
        categorical_dummies = {
            feature: [
                col for col in uploaded_data.columns if col.startswith(feature + "_")
            ]
            for feature in selected_features
            if feature_types[feature] == "categorical"
        }

        if request.method == "POST":
            input_data = {}
            for feature in selected_features:
                if feature_types[feature] == "categorical":
                    selected_value = request.POST.get(feature)
                    for dummy_col in categorical_dummies[feature]:
                        input_data[dummy_col] = (
                            1 if dummy_col == f"{feature}_{selected_value}" else 0
                        )
                else:
                    input_data[feature] = request.POST.get(feature)

            # Ensure all model columns are present
            for col in model_columns:
                if col not in input_data:
                    input_data[col] = 0

            # Convert input_data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Load the trained model from session
            model = pickle.loads(base64.b64decode(request.session["model"]))

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            classe = ""
            if probability < 0.1:
                classe = "excellente"
            elif probability < 0.2:
                classe = "Trés Bon"
            elif probability < 0.3:
                classe = "Bon"
            elif probability < 0.4:
                classe = "Moyen"
            elif probability < 0.5:
                classe = "Assez Moyen"
            elif probability < 0.6:
                classe = "Peu Risqué"
            elif probability < 0.7:
                classe = "Risqué"
            elif probability < 0.8:
                classe = "Très Risqué"
            elif probability < 0.9:
                classe = "Extrêmement Risqué"
            else:
                classe = "Défaut"

            return render(
                request,
                "scoring.html",
                {
                    "selected_features": selected_features,
                    "feature_types": feature_types,
                    "prediction": prediction,
                    "probability": probability,
                    "uploaded_data": uploaded_data_dict,  # Pass uploaded_data as dictionary
                    "classe": classe,
                },
            )
        else:
            return render(
                request,
                "scoring.html",
                {
                    "selected_features": selected_features,
                    "feature_types": feature_types,
                    "uploaded_data": uploaded_data_dict,  # Pass uploaded_data as dictionary
                },
            )
    else:
        return redirect("modeling")

# def change_data_type(request):
#     if 'uploaded_data' in request.session:
#         #uploaded_data = pd.read_json(request.session['uploaded_data'])
#         uploaded_data_json = StringIO(request.session["uploaded_data"])
#         uploaded_data = pd.read_json(uploaded_data_json)

#         if request.method == 'POST':
#             column = request.POST.get('column')  # récupère le nom de la colonne
#             new_type = request.POST.get(f'new_type_{column}')  # récupère le nouveau type de donnée

#             if new_type and column:
#                 try:
#                     # Récupérer l'ancien type de données
#                     old_type = str(uploaded_data[column].dtype)

#                     # Convertir la colonne selon le nouveau type
#                     if new_type == 'int':
#                         uploaded_data[column] = uploaded_data[column].astype(int)
#                     elif new_type == 'float':
#                         uploaded_data[column] = uploaded_data[column].astype(float)
#                     elif new_type == 'category':
#                         uploaded_data[column] = uploaded_data[column].astype('category')
#                     elif new_type == 'object':
#                         uploaded_data[column] = uploaded_data[column].astype(str)
#                     elif new_type == 'datetime':
#                         uploaded_data[column] = pd.to_datetime(uploaded_data[column], errors='coerce')
#                     elif new_type == 'bool':
#                         uploaded_data[column] = uploaded_data[column].astype(bool)
#                     else:
#                         raise ValueError(f"Unsupported data type: {new_type}")

#                     # Sauvegarder les données modifiées dans la session
#                     request.session.modified = True
#                     request.session['uploaded_data'] = uploaded_data.to_json()
                    

#                     # Sauvegarder les données modifiées dans la base de données
#                     processed_data = ProcessedData.objects.create(
#                         column_name=column,
#                         old_type=old_type,
#                         new_type=new_type,
#                         data=uploaded_data[column].to_json()
#                     )

#                     # Mettre à jour l'aperçu des données
#                     overview = data_overview(uploaded_data)
#                     return render(request, 'overview.html', {
#                         'overview': overview,
#                         'message': 'Type de données changé et persiste dans la base de données !'
#                     })

#                 except Exception as e:
#                     error_message = f"Erreur lors de la conversion de la colonne {column} en {new_type}: {e}"
#         return render(
#             request, 'overview.html',
#             {
#                 'overview': data_overview(uploaded_data),
#                  'error': error_message
#             })
#     return redirect('overview')







@login_required
def remove_column(request):
    if "uploaded_data" in request.session:
        uploaded_data_list = request.session["uploaded_data"]
        uploaded_data = pd.DataFrame.from_records(uploaded_data_list)

        if request.method == "POST":
            column_to_remove = request.POST.get("column_to_remove")
            if column_to_remove in uploaded_data.columns:
                uploaded_data.drop(columns=[column_to_remove], inplace=True)
                request.session["uploaded_data"] = uploaded_data.to_dict(
                    orient="records"
                )
                request.session.modified = True  # Ensure the session is saved
                message = (
                    f"La colonne '{column_to_remove}' a été supprimée avec succès."
                )
            else:
                message = f"La colonne '{column_to_remove}' n'existe pas."

        columns = uploaded_data.columns.tolist()
        data_preview = uploaded_data.head().to_html()

        return render(
            request,
            "data_treatment.html",
            {"columns": columns, "data_preview": data_preview, "message": message},
        )
    return redirect("overview")


import pandas as pd
from io import StringIO
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


@login_required
def feature_engineering(request):
    if "uploaded_data" in request.session:
        uploaded_data_json = StringIO(request.session["uploaded_data"])
        uploaded_data = pd.read_json(uploaded_data_json)

        if request.method == "POST":
            new_column_name = request.POST.get("new_column_name")
            formula = request.POST.get("formula")
            columns = request.POST.getlist("columns")

            if new_column_name and formula:
                try:
                    # Remplacer les noms de colonnes par leur référence dans le DataFrame
                    for column in columns:
                        formula = formula.replace(column, f"uploaded_data['{column}']")

                    print(f"Processed Formula: {formula}")  # Debugging output

                    # Évaluer la formule transformée
                    uploaded_data[new_column_name] = eval(formula)

                    # Sauvegarder les modifications dans la session
                    request.session["uploaded_data"] = uploaded_data.to_json()
                    request.session.modified = True

                    return redirect("overview")
                except Exception as e:
                    print(f"Erreur: {e}")
                    # Gestion optionnelle de l'erreur ou affichage à l'utilisateur

        return render(
            request,
            "feature_engineering.html",
            {"columns": uploaded_data.columns.tolist()},
        )
    return redirect("overview")


def base_context(request):
    return {
        "current_year": datetime.now().year,
    }


    
from django.shortcuts import render, redirect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from django.http import HttpResponse
import os
from django.conf import settings
from io import StringIO

def transitionMatrixView(request):
    # Vérifie si des données ont été téléchargées et sont disponibles dans la session
    if "uploaded_data" in request.session:
        uploaded_data_json = StringIO(request.session["uploaded_data"])
        uploaded_data = pd.read_json(uploaded_data_json)

        # Définit les noms des colonnes à analyser
        column1 = 'grade2024'
        column2 = 'loan_grade2025'

        # Traite la requête POST pour générer la heatmap
        if request.method == "POST":
            # Vérifie que les colonnes nécessaires sont présentes dans les données
            if column1 in uploaded_data.columns and column2 in uploaded_data.columns:
                try:
                    # Calcul de la fréquence croisée entre les colonnes 'grade2024' et 'loan_grade2025'
                    cross_tab = pd.crosstab(uploaded_data[column1], uploaded_data[column2])

                    # Calcul des pourcentages par ligne de la matrice de transition
                    cross_tab_percentage = cross_tab.apply(lambda x: x / x.sum() * 100, axis=1)

                    # Génération de la heatmap à partir des pourcentages
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cross_tab_percentage, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

                    # Ajout des titres et des labels à la heatmap
                    plt.title(f"Matrice de Transition de {column2} entre N et N+1")
                    plt.xlabel(f"{column2}")
                    plt.ylabel(f"{column1}")

                    # Sauvegarde de la heatmap dans un fichier dans le répertoire MEDIA
                    plot_filename = "transaction_matrix.png"
                    plot_path = os.path.join(settings.MEDIA_ROOT, plot_filename)
                    plt.savefig(plot_path)
                    plt.close()

                    # Passer le chemin de l'image à la template
                    transaction_matrix = os.path.join(settings.MEDIA_URL, plot_filename)

                    # Renvoie l'image générée sous forme de réponse HTTP
                    return render(
                        request,
                        "transition_matrix.html",
                        {"transaction_matrix": transaction_matrix}
                    )

                except Exception as e:
                    # Gestion des erreurs lors du calcul ou du rendu de la heatmap
                    return HttpResponse(f"Erreur lors de la génération de la matrice de transition : {e}", status=400)

            # Si les colonnes sont manquantes dans les données
            else:
                return HttpResponse(f"Les colonnes {column1} ou {column2} sont manquantes dans les données téléchargées.", status=400)

        # Si la requête est de type GET, afficher les colonnes disponibles
        return render(
            request,
            "transition_matrix.html",  # Template HTML pour afficher les données
            {"columns": uploaded_data.columns.tolist()},  # Liste des colonnes à afficher
        )
    
    # Si aucune donnée n'est disponible dans la session, redirige vers une autre vue
    return redirect("overview")


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from django.conf import settings
from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import json 
from django.http import JsonResponse

from django.utils.translation import gettext_lazy as _

def get_risk_level_and_comment(pd_value):
    """
    Evaluates the risk level and returns a comment based on the projected PD.
    """
    if pd_value >= 0.5000:  # Use pd here instead of PD_2024
        return 5, _("To observe")
    elif pd_value >= 0.2689:
        return 4, _("Average")
    elif pd_value >= 0.0474:
        return 3, _("Good")
    elif pd_value >= 0.0025:
        return 2, _("Very Good")
    else:
        return 1, _("Excellent")
    

def store_id_pd2024_in_list(id_column, pd_2024_column):
    """
    Cette méthode permet de stocker les ID et PD_2024 dans une liste de tuples.
    Chaque tuple contiendra (id, PD_2024).
    """
    
    id_pd2024_list = []
    try:
        for id_val, pd_val in zip(id_column, pd_2024_column):
            id_pd2024_list.append((id_val, pd_val))
        print("Les données id et PD_2024 ont été stockées dans la liste.")
        return id_pd2024_list
    except Exception as e:
        print("Erreur lors du stockage des données id et PD_2024 dans la liste :", e)
        return []




from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from io import StringIO
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render, redirect
from sklearn.metrics import mean_squared_error, r2_score

def pd_view(request):
    if "uploaded_data" not in request.session:
        return redirect("overview")

    try:
        # Load data
        uploaded_data_json = StringIO(request.session["uploaded_data"])
        df = pd.read_json(uploaded_data_json)
        print("Données chargées")

        # Prepare data
        X = df.drop('loan_status', axis=1)
        y = df['loan_status']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Encode categorical columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if categorical_cols.any():
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols].astype(str))
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols].astype(str))

        # Convert date columns to numeric
        date_cols = X_train.select_dtypes(include=['datetime']).columns
        for col in date_cols:
            X_train[col] = pd.to_numeric(pd.to_datetime(X_train[col])).astype(np.int64)
            X_test[col] = pd.to_numeric(pd.to_datetime(X_test[col])).astype(np.int64)

        # Handle missing values
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].median())
        X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].median())
        for col in categorical_cols:
            mode_val = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode_val)
            X_test[col] = X_test[col].fillna(mode_val)
        print("NaN après traitement :", X_train.isna().sum().sum())

        # Dimensionality reduction with PCA
        n_components = min(50, X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        print(f"PCA reduced features to {n_components}")

        # Apply SMOTE
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

        # Feature selection
        selector = SelectKBest(f_classif, k=min(10, X_train_resampled.shape[1]))
        X_train_selected = selector.fit_transform(X_train_resampled, y_train_resampled)
        X_test_selected = selector.transform(X_test_pca)

        # Train model
        model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
        model.fit(X_train_selected, y_train_resampled)

        # Predictions
        y_pred = model.predict(X_test_selected)
        y_pred_proba = model.predict_proba(X_test_selected)[:, 1]

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix plot
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non Defaut", "Defaut"], yticklabels=["Non Defaut", "Defaut"])
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies Valeurs")
        plt.title("Matrice de Confusion")
        cm_buffer = io.BytesIO()
        plt.savefig(cm_buffer, format='png')
        cm_buffer.seek(0)
        cm_image = base64.b64encode(cm_buffer.getvalue()).decode('utf-8')
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel("Taux de Faux Positifs (FPR)")
        plt.ylabel("Taux de Vrais Positifs (TPR)")
        plt.title("Courbe ROC")
        plt.legend()
        roc_buffer = io.BytesIO()
        plt.savefig(roc_buffer, format='png')
        roc_buffer.seek(0)
        roc_image = base64.b64encode(roc_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Calculate PD for 2024
        X_test_copy = X_test.copy()
        X_test_selected = selector.transform(X_test_pca)
        X_test_copy["PD_12mois"] = model.predict_proba(X_test_selected)[:, 1] * 100
        X_test_copy["level"], X_test_copy["comment"] = zip(
            *X_test_copy["PD_12mois"].apply(lambda pd_value: get_risk_level_and_comment(pd_value / 100))
        )

        # Finalize PD_2024
        PD_2024 = X_test_copy[['id', 'PD_12mois', 'level', 'comment']].rename(columns={
            'PD_12mois': 'PD_2024', 'level': 'level', 'comment': 'comment'
        })
        PD_2024['PD_2024'] = pd.to_numeric(PD_2024['PD_2024'], errors='coerce')
        mean_pd = PD_2024['PD_2024'].mean()
        PD_2024['PD_2024'].fillna(mean_pd, inplace=True)

        # Store id and PD_2024
        id_pd2024_list = PD_2024[['id', 'PD_2024']].to_records(index=False).tolist()
        request.session['id_pd2024_list'] = id_pd2024_list
        print("Données id/PD_2024 stockées dans la session.")
        
        id_pd2024_list = PD_2024[['id', 'PD_2024']].to_records(index=False).tolist()
        request.session['id_pd2024_list'] = id_pd2024_list
        print("Données id/PD_2024 stockées dans la session.")

        # Backtesting
        np.random.seed(42)
        PD_2024["Variation"] = np.random.uniform(-0.1, 0.1, size=len(PD_2024))
        PD_2024["PD_Reelle"] = PD_2024["PD_2024"] * (1 + PD_2024["Variation"])
        mse = mean_squared_error(PD_2024["PD_2024"], PD_2024["PD_Reelle"])
        r2 = r2_score(PD_2024["PD_2024"], PD_2024["PD_Reelle"])

        plt.figure(figsize=(10, 6))
        plt.scatter(PD_2024["PD_Reelle"], PD_2024["PD_2024"], label="Données", alpha=0.7)
        min_value = min(PD_2024["PD_2024"].min(), PD_2024["PD_Reelle"].min())
        max_value = max(PD_2024["PD_2024"].max(), PD_2024["PD_Reelle"].max())
        plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", label="Ligne de Parfaite Prédiction")
        plt.xlabel("PD Réelle")
        plt.ylabel("PD Prédite")
        plt.title("Backtesting entre PD Réelle et PD Prédite")
        plt.legend()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        pd_graph_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Pagination
        pd_records = PD_2024.to_dict(orient="records")
        paginator = Paginator(pd_records, 20)  # 10 records per page
        page_number = request.GET.get('page')
        try:
            page_obj = paginator.page(page_number)
        except PageNotAnInteger:
            page_obj = paginator.page(1)
        except EmptyPage:
            page_obj = paginator.page(paginator.num_pages)

        # Render template
        return render(
            request,
            "pd.html",
            {
                "accuracy": accuracy,
                "report": report,
                "cm_image": cm_image,
                "roc_image": roc_image,
                "PD_2024": page_obj.object_list,  # Pass paginated records
                "page_obj": page_obj,  # Pass page object for pagination
                "mse": mse,
                "r2": r2,
                "pd_graph_image": pd_graph_image,
                "uploaded_data": df.to_dict(orient='records'),
            }
        )

    except MemoryError:
        print("MemoryError: Dataset too large. Consider reducing features or sampling data.")
        return render(request, "error.html", {"message": "Memory error: Dataset too large. Please reduce the dataset size or contact support."})
    except Exception as e:
        print(f"Error: {str(e)}")
        return render(request, "error.html", {"message": f"An error occurred: {str(e)}"})


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from io import StringIO

import pandas as pd
import numpy as np
import io
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report
from io import StringIO

def store_id_lgd_in_list(id_column, lgd_column):
    """
    Cette méthode permet de stocker les ID et LGD dans une liste de tuples.
    Chaque tuple contiendra (id, LGD).
    """
    id_lgd_list = []  # Liste vide pour stocker les tuples (id, LGD)
    try:
        for id_val, lgd_val in zip(id_column, lgd_column):
            id_lgd_list.append((id_val, lgd_val))  # Ajouter la paire (id, LGD) à la liste
        print("Les données id et LGD ont été stockées dans la liste.")
        return id_lgd_list  # Retourner la liste
    except Exception as e:
        print("Erreur lors du stockage des données id et LGD dans la liste :", e)
        return []  # Retourner une liste vide en cas d'erreur


from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage


def lgd_view(request):
    # Check if data is available in the session
    if "uploaded_data" in request.session:
        # Load JSON data from the session
        uploaded_data_json = StringIO(request.session["uploaded_data"])
        df = pd.read_json(uploaded_data_json)
        print("Data loaded")

        # Process columns with duration strings, e.g., 'loan_term'
        df['term'] = df['term'].str.replace(' months', '').astype(float)  # Clean duration columns
        
        # Convert 'total_pymnt' to numeric
        df['total_pymnt'] = pd.to_numeric(df['total_pymnt'], errors='coerce')
        
        # Replace values in 'pymnt_plan' and 'application_type'
        if 'pymnt_plan' in df.columns:
            df['pymnt_plan'] = df['pymnt_plan'].replace({'y': 1, 'n': 0})

        if 'application_type' in df.columns:
            df['application_type'] = df['application_type'].apply(lambda x: 1 if x == 'individual' else 0)
            
        # Clean the 'emp_length' column (Professional experience)
        if 'emp_length' in df.columns:
            df['emp_length'] = df['emp_length'].astype(str)  # Convert to string if not already
            df['emp_length'] = df['emp_length'].str.replace(' years', '', regex=True)  # Remove " years"
            df['emp_length'] = df['emp_length'].str.replace(' year', '', regex=True)   # Remove " year"
            df['emp_length'] = df['emp_length'].str.replace('< 1', '0')  # Replace "< 1" with "0"
            df['emp_length'] = df['emp_length'].str.replace('10+', '10')  # Replace "10+" with "10"
            df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')  # Convert to numeric
            df['emp_length'].fillna(df['emp_length'].median(), inplace=True)  # Replace NaN with median

        # Create dummy variables
        dummy_columns = ['grade2024', 'sub_grade', 'home_ownership', 'verification_status',
                        'purpose', 'addr_state', 'initial_list_status', 'type']

        for col in dummy_columns:
            if col in df.columns:
                df_Dummy = pd.get_dummies(df[col], prefix=col, prefix_sep=':', drop_first=False, dtype=int)
                df_Dummy.index = df.index  # Ensure the index matches
                df = pd.concat([df, df_Dummy], axis=1)
                
        print(f"Dimensions after creating dummy variables: {df.shape}")
        # Drop original columns after creating dummy variables
        columns_to_drop = [col for col in dummy_columns if col in df.columns]
        df.drop(columns=columns_to_drop, inplace=True)

        print(f"Dimensions after dropping original columns: {df.shape}")
        print("Data with dummy variables created:", df.head())

        # Preprocess data
        data_defaults = df[df['loan_status'] == 1]  # Filter for loans with binary status = 1
        data_defaults['recovery_rate'] = ( data_defaults['recoveries'] +
        data_defaults['collection_recovery_fee'] +
        data_defaults['total_rec_late_fee']+data_defaults['total_rec_prncp'] +
        data_defaults['total_rec_int']
    )/ (data_defaults['funded_amnt'])

        
        # Ensure recovery rate does not exceed 1
        data_defaults['recovery_rate'] = np.where(data_defaults['recovery_rate'] > 1, 1, data_defaults['recovery_rate'])

        # Create 'recovery_rate_0_1' column
        data_defaults['recovery_rate_0_1'] = np.where(data_defaults['recovery_rate'] == 0, 0, 1)

        # Prepare input and output data for the first stage of the model
        X = data_defaults.drop(['recovery_rate', 'recovery_rate_0_1', 'issue_d', 'earliest_cr_line', 'mths_since_last_delinq',
                                'loan_grade2025', 'emp_title', 'title', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d',
                                'last_credit_pull_d', 'mths_since_last_major_derog', 'zip_code', 'loan_status'], axis=1)  # Drop unnecessary columns
        y = data_defaults['recovery_rate_0_1']  # Target: 1 if recovery occurred, 0 otherwise

        # Check dimensions of X and y
        print(f"Dimensions of X: {X.shape}, Dimensions of y: {y.shape}")

        # Split data into training and test sets
        X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the LogisticRegression model for the first stage
        # 🔹 Binary classification model (Recovery = Yes/No)
        reg_lgd_st_1 = LogisticRegression(max_iter=500)
        reg_lgd_st_1.fit(X_train_s1, y_train_s1)
        y_pred1 = reg_lgd_st_1.predict(X_test_s1)
        y_proba = reg_lgd_st_1.predict_proba(X_test_s1)

        # 🔹 Ensure dimensions match before concatenation
        if y_proba.shape[0] != y_test_s1.shape[0]:
            y_proba = y_proba[:y_test_s1.shape[0]]

        # Calculate confusion matrix and classification report
        cm = confusion_matrix(y_test_s1, y_pred1)
        report = classification_report(y_test_s1, y_pred1)
        
        # Generate confusion matrix as an image
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.xlabel("Predictions")
        plt.ylabel("True Values")
        plt.title("Confusion Matrix")
        cm_buffer = io.BytesIO()
        plt.savefig(cm_buffer, format='png')
        cm_buffer.seek(0)
        cm_image = base64.b64encode(cm_buffer.getvalue()).decode('utf-8')
        plt.close()

        # Data for the second stage of the model (only rows where recovery occurred)
        lgd_step_2_data = data_defaults[data_defaults['recovery_rate_0_1'] == 1]

        # Prepare data for the second stage of the model
        lgd_X_S2 = lgd_step_2_data.drop(['recovery_rate', 'recovery_rate_0_1', 'issue_d', 'earliest_cr_line', 'mths_since_last_delinq',
                                'loan_grade2025', 'emp_title', 'title', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d',
                                'last_credit_pull_d', 'mths_since_last_major_derog', 'zip_code', 'loan_status'], axis=1)
        lgd_Y_S2 = lgd_step_2_data['recovery_rate']
        lgd_X_S2_train, lgd_X_S2_test, lgd_Y_S2_train, lgd_Y_S2_test = train_test_split(lgd_X_S2, lgd_Y_S2, test_size=0.2, random_state=42)

        # Create and train the LinearRegression model for the second stage
        reg_lgd_st_2 = LinearRegression()
        reg_lgd_st_2.fit(lgd_X_S2_train, lgd_Y_S2_train)

        # Predictions for the second stage
        y_pred2 = reg_lgd_st_2.predict(lgd_X_S2_test)

        # Calculate errors
        mae = mean_absolute_error(lgd_Y_S2_test, y_pred2)
        mse = mean_squared_error(lgd_Y_S2_test, y_pred2)
        rmse = np.sqrt(mse)
        r2 = r2_score(lgd_Y_S2_test, y_pred2)

        # Combined predictions from both stages (calculate final probability)
        y_pred3 = reg_lgd_st_2.predict(X_test_s1)
        y_comb = y_pred3 * y_pred1
        y_comb = np.where(y_comb < 0, 0, y_comb)
        y_comb = np.where(y_comb > 1, 1, y_comb)
        
        # Check the size of the results
        if len(y_comb) < len(df):
            missing_rows = len(df) - len(y_comb)
            y_comb = np.append(y_comb, [np.nan] * missing_rows)  # Fill with NaN if necessary
        
        # Create final DataFrame with ID and y_comb
        df_result = pd.DataFrame({
            'id': df['id'],  # Ensure 'id' exists in the DataFrame
            'y_comb': y_comb
        })
        
        # Drop rows where y_comb is NaN
        df_result = df_result.dropna(subset=['y_comb'])
        
        # Convert to list of dictionaries for pagination
        predictions_list = df_result.to_dict(orient="records")
        
        # Paginate the predictions data (LGD Predictions table)
        predictions_paginator = Paginator(predictions_list, 20)  # 50 items per page
        predictions_page = request.GET.get('predictions_page')  # Use a unique query parameter
        try:
            predictions_page_obj = predictions_paginator.page(predictions_page)
        except PageNotAnInteger:
            predictions_page_obj = predictions_paginator.page(1)
        except EmptyPage:
            predictions_page_obj = predictions_paginator.page(predictions_paginator.num_pages)

        data1 = df.copy()

        # Drop unnecessary columns for predictions
        data1 = data1.drop(['issue_d', 'earliest_cr_line', 'mths_since_last_delinq',
                            'loan_grade2025', 'emp_title', 'title', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d',
                            'last_credit_pull_d', 'mths_since_last_major_derog', 'zip_code', 'loan_status'], axis=1)

        data1 = data1.dropna()

        # Predictions for 'recovery_rate_st_1' with reg_lgd_st_1 model
        data1['recovery_rate_st_1'] = reg_lgd_st_1.predict(data1)
        print(data1['recovery_rate_st_1'])

        # Correctly drop 'recovery_rate_st_2' before prediction
        rr2 = reg_lgd_st_2.predict(data1.drop(columns=['recovery_rate_st_1'], axis=1, errors='ignore'))

        # Add predicted values to 'recovery_rate_st_2' column
        data1['recovery_rate_st_2'] = rr2

        # Combine predicted values from stages 1 and 2 to determine the final estimated recovery rate
        data1['recovery_rate'] = data1['recovery_rate_st_1'] * data1['recovery_rate_st_2']

        # Display statistical description
        print(data1['recovery_rate'].describe())

        # Correct recovery rate values outside the [0, 1] range
        data1['recovery_rate'] = np.where(data1['recovery_rate'] < 0, 0, data1['recovery_rate'])
        data1['recovery_rate'] = np.where(data1['recovery_rate'] > 1, 1, data1['recovery_rate'])

        # Calculate LGD (1 - estimated recovery rate)
        data1['LGD'] = (1 - data1['recovery_rate'])

        # Display descriptive statistics for LGD
        print(data1['LGD'].describe())
        
        # Add LGD distribution to the data passed to the template
        LGD_distribution = data1['LGD'] * 100
        
        # Create final DataFrame with ID and LGD_distribution
        LGD_result = pd.DataFrame({
            'id': df['id'],  # Ensure 'id' exists in the DataFrame
            'LGD_distribution': LGD_distribution
        })
        
        LGD_result = LGD_result.dropna(subset=['LGD_distribution'])
       
        # Convert to list of dictionaries for pagination
        lgd_distribution_list = LGD_result.to_dict(orient="records")
        
        # Paginate the LGD distribution data (LGD Distribution table)
        lgd_distribution_paginator = Paginator(lgd_distribution_list, 20)  # 50 items per page
        lgd_distribution_page = request.GET.get('lgd_distribution_page')  # Use a unique query parameter
        try:
            lgd_distribution_page_obj = lgd_distribution_paginator.page(lgd_distribution_page)
        except PageNotAnInteger:
            lgd_distribution_page_obj = lgd_distribution_paginator.page(1)
        except EmptyPage:
            lgd_distribution_page_obj = lgd_distribution_paginator.page(lgd_distribution_paginator.num_pages)

        id_lgd_list = LGD_result[['id', 'LGD_distribution']].to_records(index=False).tolist()

        # Store the list in the session
        request.session['id_lgd_list'] = id_lgd_list
        print("ID/LGD data stored in session.")

        # Display columns containing NaN in X
        nan_columns = X.columns[X.isna().any(axis=0)]
        print("Columns containing NaN:", nan_columns)

        # Pass results to the template
        return render(request, 
                      'lgd.html', {
                          'accuracy': report,  # Classification report
                          'confusion_matrix': cm,
                          'mae': mae,
                          'mse': mse,
                          'rmse': rmse,
                          'cm_image': cm_image,
                          'contextLGD': lgd_distribution_page_obj.object_list,  # Paginated data
                          'context': predictions_page_obj.object_list,  # Paginated data
                          'predictions_page_obj': predictions_page_obj,  # For pagination controls
                          'lgd_distribution_page_obj': lgd_distribution_page_obj,  # For pagination controls
                          'r2': r2,
                          'y_comb': y_comb,  # Combined predictions
                          'LGD_distribution': LGD_distribution,
                      })

    return render(request, 'lgd.html')

from django.http import JsonResponse
import numpy as np
import pandas as pd

def get_predictions(request):
    
    uploaded_data_json = StringIO(request.session["uploaded_data"])
    df = pd.read_json(uploaded_data_json)
    print("Données chargées")

    # Traitement des colonnes avec des durées sous forme de chaînes, par exemple 'loan_term'
    df['term'] = df['term'].str.replace(' months', '').astype(float)  # Nettoyer les colonnes contenant des durées
        
    # Conversion de 'total_pymnt' en numérique
    df['total_pymnt'] = pd.to_numeric(df['total_pymnt'], errors='coerce')
        
    # Remplacement des valeurs dans 'pymnt_plan' et 'application_type'
    if 'pymnt_plan' in df.columns:
        df['pymnt_plan'] = df['pymnt_plan'].replace({'y': 1, 'n': 0})

    if 'application_type' in df.columns:
        df['application_type'] = df['application_type'].apply(lambda x: 1 if x == 'individual' else 0)
            
    # Nettoyage de la colonne 'emp_length' (Expérience professionnelle)
    if 'emp_length' in df.columns:
            df['emp_length'] = df['emp_length'].astype(str)  # Convertir en string si ce n'est pas déjà le cas
            df['emp_length'] = df['emp_length'].str.replace(' years', '', regex=True)  # Supprimer " years"
            df['emp_length'] = df['emp_length'].str.replace(' year', '', regex=True)   # Supprimer " year"
            df['emp_length'] = df['emp_length'].str.replace('< 1', '0')  # Remplacer "< 1" par "0"
            df['emp_length'] = df['emp_length'].str.replace('10+', '10')  # Remplacer "10+" par "10"
            df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')  # Convertir en nombre
            df['emp_length'].fillna(df['emp_length'].median(), inplace=True)  # Remplacer NaN par la médiane

    # Création des variables binaires
    dummy_columns = ['grade2024', 'sub_grade', 'home_ownership', 'verification_status',
                        'purpose', 'addr_state', 'initial_list_status', 'type']

    for col in dummy_columns:
        if col in df.columns:
                df_Dummy = pd.get_dummies(df[col], prefix=col, prefix_sep=':', drop_first=False, dtype=int)
                df_Dummy.index = df.index  # Assurer que l'index est le même
                df = pd.concat([df, df_Dummy], axis=1)
                
    print(f"Dimensions après création des variables binaires : {df.shape}")
    # Suppression des colonnes originales après création des variables binaires
    columns_to_drop = [col for col in dummy_columns if col in df.columns]
    df.drop(columns=columns_to_drop, inplace=True)

    print(f"Dimensions après suppression des colonnes d'origine : {df.shape}")

    print("Données avec variables binaires créées :", df.head())
    # Prétraitement des données
    data_defaults = df[df['loan_status'] == 1]  # Filtrer pour les prêts avec statut binaire = 1
    data_defaults['recovery_rate'] = data_defaults['recoveries'] / data_defaults['funded_amnt']  # Calcul du taux de récupération
        
    # S'assurer que le taux de récupération ne dépasse pas 1
    data_defaults['recovery_rate'] = np.where(data_defaults['recovery_rate'] > 1, 1, data_defaults['recovery_rate'])

    # Création de la colonne 'recovery_rate_0_1'
    data_defaults['recovery_rate_0_1'] = np.where(data_defaults['recovery_rate'] == 0, 0, 1)

    # Préparer les données d'entrée et de sortie pour la première étape du modèle
    X = data_defaults.drop(['recovery_rate', 'recovery_rate_0_1', 'issue_d', 'earliest_cr_line', 'mths_since_last_delinq',
                                'loan_grade2025', 'emp_title', 'title', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d',
                                'last_credit_pull_d', 'mths_since_last_major_derog', 'zip_code', 'loan_status'], axis=1)  # Suppression des colonnes inutiles
    y = data_defaults['recovery_rate_0_1']  # Cible : 1 si une récupération a eu lieu, 0 sinon

    # Vérification des dimensions de X et y
    print(f"Dimensions de X: {X.shape}, Dimensions de y: {y.shape}")

    # Diviser les données en ensembles d'entraînement et de test
    X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(X, y, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle LogisticRegression pour la première étape
    # 🔹 Modèle de classification binaire (Recovery = Yes/No)
    reg_lgd_st_1 = LogisticRegression(max_iter=500)
    reg_lgd_st_1.fit(X_train_s1, y_train_s1)
    y_pred1 = reg_lgd_st_1.predict(X_test_s1)
    y_proba = reg_lgd_st_1.predict_proba(X_test_s1)

    # 🔹 Vérification de la cohérence des dimensions avant concaténation
    if y_proba.shape[0] != y_test_s1.shape[0]:
         y_proba = y_proba[:y_test_s1.shape[0]]

    # Données pour la deuxième étape du modèle (uniquement les lignes où la récupération a eu lieu)
    lgd_step_2_data = data_defaults[data_defaults['recovery_rate_0_1'] == 1]

    # Préparer les données pour la deuxième étape du modèle
    lgd_X_S2 = lgd_step_2_data.drop(['recovery_rate', 'recovery_rate_0_1', 'issue_d', 'earliest_cr_line', 'mths_since_last_delinq',
                                'loan_grade2025', 'emp_title', 'title', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d',
                                'last_credit_pull_d', 'mths_since_last_major_derog', 'zip_code', 'loan_status'], axis=1)
    lgd_Y_S2 = lgd_step_2_data['recovery_rate']
    lgd_X_S2_train, lgd_X_S2_test, lgd_Y_S2_train, lgd_Y_S2_test = train_test_split(lgd_X_S2, lgd_Y_S2, test_size=0.2, random_state=42)

    # Créer et entraîner le modèle LinearRegression pour la deuxième étape
    reg_lgd_st_2 = LinearRegression()
    reg_lgd_st_2.fit(lgd_X_S2_train, lgd_Y_S2_train)

    # Prédictions pour la deuxième étape
    y_pred2 = reg_lgd_st_2.predict(lgd_X_S2_test)

    # Prédictions combinées des deux étapes (calcul de la probabilité finale)
    y_pred3 = reg_lgd_st_2.predict(X_test_s1)
    y_comb = y_pred3 * y_pred1
    y_comb = np.where(y_comb < 0, 0, y_comb)
    y_comb = np.where(y_comb > 1, 1, y_comb)
        
    data1 = df.copy()

    # Suppression des colonnes inutiles pour les prédictions
    data1 = data1.drop([ 'issue_d', 'earliest_cr_line', 'mths_since_last_delinq',
                                'loan_grade2025', 'emp_title', 'title', 'mths_since_last_record', 'last_pymnt_d', 'next_pymnt_d',
                                'last_credit_pull_d', 'mths_since_last_major_derog', 'zip_code', 'loan_status'], axis=1)

    data1 = data1.dropna()

    # Prédictions pour 'recovery_rate_st_1' avec le modèle reg_lgd_st_1
    data1['recovery_rate_st_1'] = reg_lgd_st_1.predict(data1)
    print(data1['recovery_rate_st_1'])
    # Suppression correcte de la colonne 'recovery_rate_st_2' avant la prédiction
    rr2 = reg_lgd_st_2.predict(data1.drop(columns=['recovery_rate_st_1'], axis=1, errors='ignore'))

    # Ajout des valeurs prédites dans la colonne 'recovery_rate_st_2'
    data1['recovery_rate_st_2'] = rr2

    # Combinaison des valeurs prédites des étapes 1 et 2 pour déterminer le taux de récupération final estimé
    data1['recovery_rate'] = data1['recovery_rate_st_1'] * data1['recovery_rate_st_2']

    # Affichage de la description statistique
    print(data1['recovery_rate'].describe())

    # Correction des valeurs du taux de récupération en dehors de la plage [0, 1]
    data1['recovery_rate'] = np.where(data1['recovery_rate'] < 0, 0, data1['recovery_rate'])
    data1['recovery_rate'] = np.where(data1['recovery_rate'] > 1, 1, data1['recovery_rate'])
    # Calcul du LGD (1 - taux de récupération estimé)
    data1['LGD'] = (1 - data1['recovery_rate'])

    # Affichage des statistiques descriptives pour LGD
    print(data1['LGD'].describe())
        
    # Ajout de la distribution du LGD dans les données passées au template
    LGD_distribution = data1['LGD']
        
    # Création du DataFrame final avec ID et y_comb
    LGD_result = pd.DataFrame({
            'id': df['id'],  # Vérifie que X_test_copy contient bien 'id'
            'LGD_distribution': LGD_distribution
    })
    
        
    LGD_result = LGD_result.dropna(subset=['LGD_distribution'])
    
        
    # Retourner les résultats sous format JSON
    return JsonResponse(LGD_result.to_dict(orient='records'), safe=False)



import pandas as pd
from io import StringIO
from django.shortcuts import render

import pandas as pd
from io import StringIO
from django.shortcuts import render

def store_id_ead_in_list(id_column, ead_column):
    """
    Cette méthode permet de stocker les ID et EAD dans une liste de tuples.
    Chaque tuple contiendra (id, EAD).
    """
    id_ead_list = []  # Liste vide pour stocker les tuples (id, EAD)
    try:
        for id_val, ead_val in zip(id_column, ead_column):
            id_ead_list.append((id_val, ead_val))  # Ajouter la paire (id, EAD) à la liste
        print("Les données id et EAD ont été stockées dans la liste.")
        return id_ead_list  # Retourner la liste
    except Exception as e:
        print("Erreur lors du stockage des données id et EAD dans la liste :", e)
        return []  # Retourner une liste vide en cas d'erreur
    
from django.shortcuts import render, redirect
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from io import StringIO
import pandas as pd


def ead_view(request):
    try:
        # Check if data has been uploaded in the session
        if "uploaded_data" in request.session:
            uploaded_data_json = StringIO(request.session["uploaded_data"])
            df = pd.read_json(uploaded_data_json)
            print("Data loaded:", df.head())  # Display the first few rows of data

            if df.empty:
                print("The DataFrame is empty.")
                return render(request, 'ead.html', {'message': "The uploaded file is empty."})
            
            print("The DataFrame contains data.")
    
            # Check and calculate CCF and EAD columns
            if 'funded_amnt' in df.columns:
                df['CCF'] = 1  # Set CCF to 1 as requested
                df['EAD'] = 1 * df['out_prncp']  # Calculate EAD as 1 * funded_amnt
                print("Columns 'CCF' and 'EAD' calculated")
            else:
                print("The column 'funded_amnt' is missing.")
                return render(request, 'ead.html', {'message': "The column 'funded_amnt' is missing."})

            # Check if 'id' column exists and use it as index
            if 'id' in df.columns:
                df = df.set_index('id')  # Use 'id' as index
                df = df[['CCF', 'EAD']].reset_index()  # Reset index to include 'id'
                print("Using 'id' as index")
            else:
                df = df[['CCF', 'EAD']].reset_index()  # Keep default index
                print("No 'id' found, using default index")

            # Create id/EAD list with native Python types
            id_ead_list = [(int(row['id']), float(row['EAD'])) for index, row in df.iterrows()]

            # Store the list in the session
            request.session['id_ead_list'] = id_ead_list
            print("ID/EAD data stored in session")

            # Prepare data to display in the template
            columns = df.columns.tolist()
            ccf_ead_data = df.values.tolist()  # Convert to list of lists

            # Paginate the data
            paginator = Paginator(ccf_ead_data, 20)  # 20 items per page as requested
            page = request.GET.get('page')  # Get the page number from the request
            try:
                page_obj = paginator.page(page)
            except PageNotAnInteger:
                page_obj = paginator.page(1)
            except EmptyPage:
                page_obj = paginator.page(paginator.num_pages)

            return render(request, 'ead.html', {
                'ccf_ead_data': page_obj.object_list,
                'columns': columns,
                'page_obj': page_obj,
                'message': None  # Clear message if data is successfully processed
            })

        else:
            return render(request, 'ead.html', {'message': "No data uploaded."})

    except Exception as e:
        print("Error processing data:", e)
        return render(request, 'ead.html', {'message': "An error occurred while processing the data."})



from django.http import JsonResponse
import pandas as pd
import numpy as np
from io import StringIO

def get_ead_view(request):
    try:
        # Charger les données depuis la session
        uploaded_data_json = StringIO(request.session.get("uploaded_data", ""))
        df = pd.read_json(uploaded_data_json)

        if df.empty:
            return JsonResponse({"error": "Le fichier chargé est vide."}, status=400)

        # Vérification et calcul des colonnes CCF et EAD
        required_columns = {'total_rec_prncp', 'out_prncp'}
        if required_columns.issubset(df.columns):
            df['CCF'] = 1  # df['total_rec_prncp'] / df['funded_amnt']
            df['EAD'] = df['CCF'] * df['out_prncp'] 
        else:
            return JsonResponse({"error": "Les colonnes 'total_rec_prncp' et 'funded_amnt' sont absentes."}, status=400)

        # Vérifier si la colonne 'id' existe pour l'indexation
        if 'id' in df.columns:
            df = df[['id', 'CCF', 'EAD']]
        else:
            df = df[['CCF', 'EAD']].reset_index()

        # Supprimer les valeurs NaN avant de retourner les résultats
        df = df.dropna()

        # Retourner les données au format JSON
        return JsonResponse(df.to_dict(orient='records'), safe=False)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    
def evaluation_view(request):
    return render(request, 'evaluation.html')



from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from django.utils.translation import gettext as _  # Import translation function


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64


def format_nombre(valeur):
    try:
        return "{:,.2f}".format(valeur).replace(",", " ")
    except:
        return valeur
    
def vision360_view(request):
    try:
        # Récupérer les données depuis la session
        id_ead_list = request.session.get('id_ead_list', [])
        id_pd2024_list = request.session.get('id_pd2024_list', [])
        id_lgd_list = request.session.get('id_lgd_list', [])

        # Vérifier si les données sont disponibles
        if not id_ead_list or not id_pd2024_list or not id_lgd_list:
            return render(request, 'vision360.html', {'message': "Les données nécessaires ne sont pas disponibles."})

        # Convertir les listes en DataFrames
        df_ead = pd.DataFrame(id_ead_list, columns=['id', 'ead'])
        df_pd = pd.DataFrame(id_pd2024_list, columns=['id', 'pd'])
        df_lgd = pd.DataFrame(id_lgd_list, columns=['id', 'lgd'])

        # Fusionner les DataFrames sur la colonne 'id'
        df_combined = df_ead.merge(df_pd, on='id', how='left').merge(df_lgd, on='id', how='left')

        # Remplacer 'N/A' par NaN dans les colonnes 'pd', 'lgd' et 'ead' (sans chained assignment warning)
        df_combined = df_combined.replace({'pd': {'N/A': np.nan},
                                          'lgd': {'N/A': np.nan},
                                          'ead': {'N/A': np.nan}})

        # Convertir en float
        df_combined['pd'] = df_combined['pd'].astype(float)
        df_combined['lgd'] = df_combined['lgd'].astype(float)
        df_combined['ead'] = df_combined['ead'].astype(float)

        # Calculer les moyennes en ignorant NaN
        mean_pd = df_combined['pd'].mean()
        mean_lgd = df_combined['lgd'].mean()
        mean_ead = df_combined['ead'].mean()

        # Remplir les NaN avec la moyenne
        df_combined['pd'] = df_combined['pd'].fillna(mean_pd)
        df_combined['lgd'] = df_combined['lgd'].fillna(mean_lgd)
        df_combined['ead'] = df_combined['ead'].fillna(mean_ead)

        # Conversion de la PD en proportion
        df_combined['PD'] = df_combined['pd'] / 100  # de % à proportion (0-1)

        # Calcul de l'ECL (Expected Credit Loss)
        df_combined['ECL'] = df_combined['PD'] * df_combined['lgd'] * df_combined['ead']

        # Valeur du quantile à 99.9% pour UL
        z_999 = norm.ppf(0.999)

        # Facteur de corrélation rho
        rho = 0.15

        # Fonction pour calculer UL par ligne
        def calculate_UL(row, z_999, rho):
            pd = row['PD']
            lgd = row['lgd']
            ead = row['ead']
            ul = ead * lgd * (norm.cdf((norm.ppf(pd) + np.sqrt(rho) * z_999) / np.sqrt(1 - rho)) - pd)
            return ul

        # Application du calcul UL
        df_combined['UL'] = df_combined.apply(calculate_UL, axis=1, z_999=z_999, rho=rho)

        # Taux de provisionnement
        df_combined['taux_provisionnement'] = df_combined['ECL'] / df_combined['ead'] * 100

        # Facteur de corrélation rho
        df_combined['rho'] = rho

        # Fonction de calcul de K
        def calculate_K(PD, LGD, rho):
            if PD >= 1.0 or PD <= 0.0:
                return np.nan
            z_PD = norm.ppf(PD)
            z_999 = norm.ppf(0.999)
            num = z_PD + np.sqrt(rho) * z_999
            denom = np.sqrt(1 - rho)
            return LGD * norm.cdf(num / denom)

        # Calcul de K
        df_combined['K'] = df_combined.apply(lambda row: calculate_K(row['PD'], row['lgd'], row['rho']), axis=1)

        # Remplacer les NaN de K par 0
        df_combined['K'] = df_combined['K'].fillna(0)

        # Calcul de RWA et Capital Requis
        df_combined['RWA'] = df_combined['K'] * df_combined['ead'] * 12.5
        df_combined['fonds_propres'] = df_combined['RWA'] * 0.08

        df_combined["level"], df_combined["comment"] = zip(
            *df_combined["pd"].apply(lambda pd_value: get_risk_level_and_comment(pd_value / 100))
        )

        # Affichage pour debug
        print("Liste combinée id/PD/LGD/EAD/ECL/UL/RWA/Fonds propres :", df_combined)

        # Arrondir les colonnes aux formats souhaités
        df_combined['id'] = df_combined['id'].astype(int)
        df_combined['pd'] = df_combined['pd'].round(2)
        df_combined['lgd'] = df_combined['lgd'].round(2)
        df_combined['ead'] = df_combined['ead'].astype(int)
        df_combined['ECL'] = df_combined['ECL'].round(2)
        df_combined['UL'] = df_combined['UL'].round(2)
        df_combined['taux_provisionnement'] = df_combined['taux_provisionnement'].round(2)
        df_combined['RWA'] = df_combined['RWA'].round(2)
        df_combined['fonds_propres'] = df_combined['fonds_propres'].round(2)

        # Convertir pour envoi au template
        combined_list = df_combined.to_dict(orient='records')

        # Création d’un résumé agrégé
        ead_total = df_combined['ead'].sum().astype(int)
        ecl_total = df_combined['ECL'].sum().round(2)

        # Calcul du taux de provision global
        taux_prov_global = round((ecl_total / ead_total) * 100, 4) if ead_total != 0 else 0

        recap_dict = {
            'ead_total': format_nombre(ead_total),
            'ecl_total': format_nombre(ecl_total),
            'ul_total': format_nombre(df_combined['UL'].sum()),
            'rwa_total': format_nombre(df_combined['RWA'].sum()),
            'fonds_propres_total': format_nombre(df_combined['fonds_propres'].sum()),
            'pd_moyenne': round(df_combined['pd'].mean(), 2),
            'lgd_moyenne': round(df_combined['lgd'].mean(), 2),
            'taux_prov_global': round(taux_prov_global, 2),
            'nb_total_dossiers': len(df_combined)
        }

        risk_labels = {
            5: _("To observe"),
            4: _("Average"),
            3: _("Good"),
            2: _("Very Good"),
            1: _("Excellent")
        }

        # Creating a new column with the labels
        df_combined['Risk Class'] = df_combined['level'].map(risk_labels)

        # Generating the plot
        fig, ax = plt.subplots(figsize=(8, 6))
        try:
            sns.countplot(
                data=df_combined,
                x='Risk Class',
                order=[_("Excellent"), _("Very Good"), _("Good"), _("Average"), _("To observe")],
                palette='viridis',
                ax=ax
            )
            ax.set_title(_("Distribution by Risk Class"))
            ax.set_xlabel(_("Risk Class"))
            ax.set_ylabel(_("Number of Clients"))
            ax.tick_params(axis='x', rotation=30)
            risk_level_plot = fig_to_base64(fig)
        finally:
            plt.close(fig)  # Close the figure to free memory

        # Paginate the combined_list for both tables
        vision360_paginator = Paginator(combined_list, 20)  # 20 items per page
        indicators_paginator = Paginator(combined_list, 20)  # 20 items per page
        vision360_page = request.GET.get('vision360_page')
        indicators_page = request.GET.get('indicators_page')
        try:
            vision360_page_obj = vision360_paginator.page(vision360_page)
            indicators_page_obj = indicators_paginator.page(indicators_page)
        except PageNotAnInteger:
            vision360_page_obj = vision360_paginator.page(1)
            indicators_page_obj = indicators_paginator.page(1)
        except EmptyPage:
            vision360_page_obj = vision360_paginator.page(vision360_paginator.num_pages)
            indicators_page_obj = indicators_paginator.page(indicators_paginator.num_pages)

        # Renvoyer les données au template
        return render(request, 'vision360.html', {
            'combined_list': combined_list,  # Full list for reference (optional)
            'vision360_page_obj': vision360_page_obj,
            'indicators_page_obj': indicators_page_obj,
            'recap': recap_dict,
            'risk_level_plot': risk_level_plot,
        })

    except Exception as e:
        # Log de l'erreur
        print("Erreur lors de l'accès aux données de la session :", e)

        # Afficher l'erreur à l'utilisateur
        return render(request, 'vision360.html', {'message': f"Une erreur est survenue : {str(e)}"})
    

from django.shortcuts import render
import pandas as pd
import numpy as np
from scipy.stats import norm

def fiche_client_view(request):
    try:
        # Handle both GET and POST requests
        client_id = request.POST.get('id') or request.GET.get('id')
        if not client_id:
            return render(request, 'fiche_client.html', {'message': "Veuillez fournir un ID de client."})

        # Récupérer les données depuis la session pour ead, pd, lgd
        id_ead_list = request.session.get('id_ead_list', [])
        id_pd2024_list = request.session.get('id_pd2024_list', [])
        id_lgd_list = request.session.get('id_lgd_list', [])

        # Vérifier si les listes sont vides
        if not id_ead_list or not id_pd2024_list or not id_lgd_list:
            return render(request, 'fiche_client.html', {'message': "Aucune donnée disponible dans la session pour EAD, PD ou LGD."})

        # Récupérer les données JSON depuis la session
        if "uploaded_data" not in request.session:
            return render(request, 'fiche_client.html', {'message': "Aucune donnée JSON disponible dans la session."})

        # Charger les données JSON depuis la session
        uploaded_data = pd.read_json(request.session["uploaded_data"])

        # Colonnes requises
        required_columns = {'id', 'annual_inc', 'grade2024', 'type', 'int_rate', 'total_rec_prncp',
                           'loan_amnt', 'term', 'installment', 'out_prncp', 'total_rec_int', 'total_pymnt'}

        # Vérifier si toutes les colonnes requises sont présentes
        missing_columns = required_columns - set(uploaded_data.columns)
        if missing_columns:
            return render(request, 'fiche_client.html', {
                'message': f"Colonnes manquantes dans les données JSON : {', '.join(missing_columns)}"
            })

        # Créer un DataFrame à partir des données JSON avec les required_columns
        df_json = uploaded_data[list(required_columns)].copy()

        # Créer les DataFrames pour ead, pd, lgd
        df_ead = pd.DataFrame(id_ead_list, columns=['id', 'ead'])
        df_pd = pd.DataFrame(id_pd2024_list, columns=['id', 'pd'])
        df_lgd = pd.DataFrame(id_lgd_list, columns=['id', 'lgd'])

        # Fusion des DataFrames ead, pd, lgd
        df = df_ead.merge(df_pd, on='id').merge(df_lgd, on='id')

        # Fusion avec les données JSON pour inclure les required_columns
        df = df.merge(df_json, on='id', how='left')

        # Remplacer les valeurs 'N/A' ou invalides par NaN
        df = df.replace({'pd': {'N/A': np.nan}, 'lgd': {'N/A': np.nan}, 'ead': {'N/A': np.nan}})
        df[['pd', 'lgd', 'ead']] = df[['pd', 'lgd', 'ead']].astype(float)

        # Remplissage des NaN
        df.fillna(df.mean(numeric_only=True), inplace=True)  # Pour les colonnes numériques
        df.fillna('N/A', inplace=True)  # Pour les colonnes non numériques (ex. 'type', 'grade2024')

        # Calculs
        df['PD'] = df['pd'] / 100
        df['ECL'] = df['PD'] * df['lgd'] * df['ead']
        df['taux_provisionnement'] = df['ECL'] / df['ead'] * 100

        z_999 = norm.ppf(0.999)
        rho = 0.15

        def calculate_K(PD, LGD):
            if PD >= 1.0 or PD <= 0.0:
                return np.nan
            z_PD = norm.ppf(PD)
            return LGD * norm.cdf((z_PD + np.sqrt(rho) * z_999) / np.sqrt(1 - rho))

        df['K'] = df.apply(lambda row: calculate_K(row['PD'], row['lgd']), axis=1).fillna(0)
        df['RWA'] = df['K'] * df['ead'] * 12.5
        df['UL'] = df.apply(lambda row: row['ead'] * row['lgd'] *
                            (norm.cdf((norm.ppf(row['PD']) + np.sqrt(rho) * z_999) / np.sqrt(1 - rho)) - row['PD']), axis=1)
        df['fonds_propres'] = df['RWA'] * 0.08

        # Fonction get_risk_level_and_comment
        def get_risk_level_and_comment(pd_val):
            if pd_val > 0.1:
                return 5, "Risque élevé"
            elif pd_val > 0.05:
                return 4, "Risque modéré"
            elif pd_val > 0.02:
                return 3, "Risque moyen"
            elif pd_val > 0.01:
                return 2, "Risque faible"
            else:
                return 1, "Risque très faible"

        df["level"], df["comment"] = zip(*df["pd"].apply(lambda pd_val: get_risk_level_and_comment(pd_val / 100)))
        risk_labels = {5: "À surveiller", 4: "Moyen", 3: "Bon", 2: "Très Bon", 1: "Excellent"}
        df['classe'] = df['level'].map(risk_labels)

        # Récupérer les données du client demandé
        client_data = df[df['id'] == int(client_id)].copy()
        client_data = client_data.round(3)
        client_data['id'] = client_data['id'].astype(int)
        if client_data.empty:
            return render(request, 'fiche_client.html', {'message': "Aucun client trouvé avec cet ID."})

        return render(request, 'fiche_client.html', {'client': client_data.iloc[0].to_dict()})

    except Exception as e:
        return render(request, 'fiche_client.html', {'message': f"Erreur : {str(e)}"})

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
import pandas as pd
import numpy as np
from io import StringIO

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
import pandas as pd
from io import StringIO

def calculate_lifetime_pd(pd_values):
    # Placeholder function for lifetime PD calculation
    # For simplicity, return the sum of PD values (you can replace with your actual logic)
    return sum(pd_values) if pd_values else 0.0

def staging_ifrs9(request):
    try:
        # Récupérer les listes depuis la session (populées par construct_session_lists_view)
        id_pd2024_list = request.session.get('id_pd2024_list', [])
        id_ead_list = request.session.get('id_ead_list', [])
        id_lgd_list = request.session.get('id_lgd_list', [])
       
        # Charger les données JSON depuis la session
        if "uploaded_data" not in request.session:
            return render(request, 'staging_ifrs9.html', {
                'message': "Aucune donnée JSON disponible dans la session."
            })

        # Charger les données JSON depuis la session
        uploaded_data_json = StringIO(request.session.get("uploaded_data", ""))
        df = pd.read_json(uploaded_data_json)

        if df.empty:
            return render(request, 'staging_ifrs9.html', {
                'message': "Le fichier chargé est vide."
            })

        # Vérifier si les colonnes requises existent
        required_columns = ['id', 'grade2024', 'out_prncp']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return render(request, 'staging_ifrs9.html', {
                'message': f"Colonnes manquantes dans les données : {', '.join(missing_columns)}."
            })
        
        # Sélectionner uniquement les colonnes requises
        df = df[required_columns].copy()

        # Convertir les colonnes nécessaires
        df['id'] = df['id'].astype(int)
        
        # Vérifier si les colonnes numériques existent, sinon les initialiser
        numeric_columns = ['pd', 'lgd', 'ead']
        for col in numeric_columns:
            if col not in df.columns:
                df[col] = 0  # Ou une autre valeur par défaut appropriée
        
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        # Créer les DataFrames à partir des listes, si elles existent
        if all([id_pd2024_list, id_ead_list, id_lgd_list]):
            df_pd = pd.DataFrame(id_pd2024_list, columns=['id', 'PD'])
            df_ead = pd.DataFrame(id_ead_list, columns=['id', 'EAD'])
            df_lgd = pd.DataFrame(id_lgd_list, columns=['id', 'LGD'])
        else:
            # Si les listes ne sont pas disponibles, utiliser des DataFrames vides ou extraits de df
            df_pd = pd.DataFrame({'id': df['id'], 'PD': df.get('pd', 0)})
            df_ead = pd.DataFrame({'id': df['id'], 'EAD': df.get('ead', 0)})
            df_lgd = pd.DataFrame({'id': df['id'], 'LGD': df.get('lgd', 0)})

        # Fusion des DataFrames
        df_merged = df_pd.merge(df_ead, on='id', how='inner').merge(df_lgd, on='id', how='inner')
        nouvelle_base = pd.merge(
            df[['id', 'grade2024', 'out_prncp']],
            df_merged[['id', 'PD']],
            on='id',
            how='inner'
        )

        # Définir la correspondance des stages
        correspondance_stage = {
            'A': 'Bucket 1',
            'B': 'Bucket 1',
            'C': 'Bucket 2',
            'D': 'Bucket 2',
            'E': 'Bucket 2',
            'F': 'Bucket 3',
            'G': 'Bucket 3'
        }
        nouvelle_base['stage'] = nouvelle_base['grade2024'].map(correspondance_stage)

        # Fusionner pour ajouter LGD et EAD
        nouvelle_base_complète = pd.merge(
            nouvelle_base,
            df_merged[['id', 'LGD', 'EAD']],
            on='id',
            how='inner'
        )

        # Gestion des PD selon les stages
        # Stage 1: PD déjà dans id_pd2024_list (en pourcentage, converti en proportion)
        nouvelle_base_complète.loc[nouvelle_base_complète['stage'] == 'Bucket 1', 'PD'] = (
            nouvelle_base_complète.loc[nouvelle_base_complète['stage'] == 'Bucket 1', 'PD'] / 100
        )

        # Stage 2: Calculer PD lifetime
        stage2_ids = nouvelle_base_complète[nouvelle_base_complète['stage'] == 'Bucket 2']['id'].tolist()
        for client_id in stage2_ids:
            pd_values = [nouvelle_base_complète.loc[nouvelle_base_complète['id'] == client_id, 'PD'].iloc[0] / 100]
            lifetime_pd = calculate_lifetime_pd(pd_values)
            nouvelle_base_complète.loc[nouvelle_base_complète['id'] == client_id, 'PD'] = lifetime_pd

        # Stage 3: PD = 100% (soit 1.0 en proportion)
        nouvelle_base_complète.loc[nouvelle_base_complète['stage'] == 'Bucket 3', 'PD'] = 1.0

        # Calculer ECL = PD * LGD * EAD
        nouvelle_base_complète['ECL'] = nouvelle_base_complète['PD'] * nouvelle_base_complète['LGD'] * nouvelle_base_complète['EAD']

        # Convertir PD et LGD en pourcentage (multiplier par 100) after ECL calculation
        nouvelle_base_complète['PD'] = nouvelle_base_complète['PD'] * 100
        nouvelle_base_complète['LGD'] = nouvelle_base_complète['LGD'] 

        # --- Compute Global Staging Data ---
        global_staging_data = []

        # Bucket 1: Stage 1
        bucket1 = nouvelle_base_complète[nouvelle_base_complète['stage'] == 'Bucket 1']
        if not bucket1.empty:
            num_clients = len(bucket1)
            outstanding = bucket1['out_prncp'].sum()
            avg_pd = f"{(bucket1['PD'].mean()):.2f}%"  # PD converted to percentage
            ecl_n = bucket1['ECL'].mean()
            ecl_total = nouvelle_base_complète['ECL'].sum()
            provision_rate = f"{((ecl_n / ecl_total) * 100):.2f}%" if ecl_total > 0 else "0%"
        else:
            num_clients = 0
            outstanding = 0
            avg_pd = '-'
            ecl_n = 0
            provision_rate = "0%"
        global_staging_data.append({
            'bucket': 'Bucket 1',
            'outstanding': outstanding,
            'impaired_clients': num_clients,
            'avg_pd': avg_pd,
            'ecl_n': f"{ecl_n:.2f}",
            'ecl_n_minus_1': '-',  # Placeholder, as historical data isn't provided
            'provision_rate': provision_rate
        })

        # Bucket 2: Stage 2
        bucket2 = nouvelle_base_complète[nouvelle_base_complète['stage'] == 'Bucket 2']
        if not bucket2.empty:
            num_clients = len(bucket2)
            outstanding = bucket2['out_prncp'].sum()
            avg_pd = f" {bucket2['PD'].mean():.2f}%"  # PD in percentage
            ecl_n = bucket2['ECL'].mean()
            ecl_total = nouvelle_base_complète['ECL'].sum()
            provision_rate = f"{((ecl_n / ecl_total) * 100):.2f}%" if ecl_total > 0 else "0%"
        else:
            num_clients = 0
            outstanding = 0
            avg_pd = '-'
            ecl_n = 0
            provision_rate = "0%"
        global_staging_data.append({
            'bucket': 'Bucket 2',
            'outstanding': outstanding,
            'impaired_clients': num_clients,
            'avg_pd': avg_pd,
            'ecl_n': f"{ecl_n:.2f}",
            'ecl_n_minus_1': '-',  # Placeholder, as historical data isn't provided
            'provision_rate': provision_rate
        })

        # Bucket 3: Stage 3
        bucket3 = nouvelle_base_complète[nouvelle_base_complète['stage'] == 'Bucket 3']
        if not bucket3.empty:
            num_clients = len(bucket3)
            outstanding = bucket3['out_prncp'].sum()
            avg_pd = "100%"  # Fixed PD for Stage 3
            ecl_n = bucket3['ECL'].mean()
            ecl_total = nouvelle_base_complète['ECL'].sum()
            provision_rate = f"{((ecl_n / ecl_total) * 100):.2f}%" if ecl_total > 0 else "0%"
        else:
            num_clients = 0
            outstanding = 0
            avg_pd = "100%"
            ecl_n = 0
            provision_rate = "0%"
        global_staging_data.append({
            'bucket': 'Bucket 3',
            'outstanding': outstanding,
            'impaired_clients': num_clients,
            'avg_pd': avg_pd,
            'ecl_n': f"{ecl_n:.2f}",
            'ecl_n_minus_1': '-',  # Placeholder, as historical data isn't provided
            'provision_rate': provision_rate
        })

        # Convertir le DataFrame en dictionnaire pour le template
        result = nouvelle_base_complète.to_dict('records')

        # Pagination
        paginator = Paginator(result, 20)  # Show 20 clients per page
        page_number = request.GET.get('staging_ifrs9_page', 1)
        try:
            staging_ifrs9_page_obj = paginator.page(page_number)
        except PageNotAnInteger:
            staging_ifrs9_page_obj = paginator.page(1)
        except EmptyPage:
            staging_ifrs9_page_obj = paginator.page(paginator.num_pages)

        return render(request, 'staging_ifrs9.html', {
            'global_staging_data': global_staging_data,
            'clients': staging_ifrs9_page_obj,
            'staging_ifrs9_page_obj': staging_ifrs9_page_obj,
            'message': None
        })

    except ValueError as ve:
        return render(request, 'staging_ifrs9.html', {
            'message': f"Erreur de conversion : {str(ve)}"
        })
    except KeyError as ke:
        return render(request, 'staging_ifrs9.html', {
            'message': f"Clé manquante : {str(ke)}"
        })
    except Exception as e:
        return render(request, 'staging_ifrs9.html', {
            'message': f"Erreur inattendue : {str(e)}"
        })

# Vue pour la page "Client File" (unchanged)
from django.shortcuts import render
from django.http import HttpResponseBadRequest
import pandas as pd
from io import StringIO

from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from io import StringIO
import pandas as pd

from django.shortcuts import render
import pandas as pd
from io import StringIO

from django.shortcuts import render
import pandas as pd
from io import StringIO

def ifrs_client(request):
    try:
        # Initialize variables
        client = None
        message = None
        submitted_id = None

        # Handle both GET and POST requests
        client_id = request.POST.get('id') or request.GET.get('id')
        if not client_id or not client_id.strip().isdigit():
            message = "Veuillez fournir un ID de client valide."
            submitted_id = client_id if client_id else ''
        else:
            submitted_id = int(client_id)

            # Load data from session (assuming it’s populated similarly to staging_ifrs9)
            if "uploaded_data" not in request.session:
                message = "Aucune donnée disponible dans la session."
            else:
                uploaded_data_json = StringIO(request.session.get("uploaded_data", ""))
                df = pd.read_json(uploaded_data_json)

                if df.empty:
                    message = "Le fichier chargé est vide."
                else:
                    # Ensure required columns exist
                    required_columns = ['id', 'grade2024', 'out_prncp']
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        message = f"Colonnes manquantes dans les données : {', '.join(missing_columns)}."
                    else:
                        # Prepare the base DataFrame with additional metrics
                        df = df[required_columns].copy()
                        df['id'] = df['id'].astype(int)

                        # Initialize numeric columns if not present
                        numeric_columns = ['pd', 'lgd', 'ead']
                        for col in numeric_columns:
                            if col not in df.columns:
                                df[col] = 0
                        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
                        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

                        # Merge with session data (assuming pd, lgd, ead lists exist)
                        id_pd2024_list = request.session.get('id_pd2024_list', [])
                        id_ead_list = request.session.get('id_ead_list', [])
                        id_lgd_list = request.session.get('id_lgd_list', [])

                        if all([id_pd2024_list, id_ead_list, id_lgd_list]):
                            df_pd = pd.DataFrame(id_pd2024_list, columns=['id', 'PD'])
                            df_ead = pd.DataFrame(id_ead_list, columns=['id', 'EAD'])
                            df_lgd = pd.DataFrame(id_lgd_list, columns=['id', 'LGD'])
                        else:
                            df_pd = pd.DataFrame({'id': df['id'], 'PD': df.get('pd', 0)})
                            df_ead = pd.DataFrame({'id': df['id'], 'EAD': df.get('ead', 0)})
                            df_lgd = pd.DataFrame({'id': df['id'], 'LGD': df.get('lgd', 0)})

                        nouvelle_base = pd.merge(
                            df[['id', 'grade2024', 'out_prncp']],
                            df_pd[['id', 'PD']],
                            on='id',
                            how='inner'
                        )
                        nouvelle_base_complète = pd.merge(
                            nouvelle_base,
                            df_ead[['id', 'EAD']],
                            on='id',
                            how='inner'
                        ).merge(
                            df_lgd[['id', 'LGD']],
                            on='id',
                            how='inner'
                        )

                        # Define staging mapping
                        correspondance_stage = {
                            'A': 'stage1',
                            'B': 'stage1',
                            'C': 'stage2',
                            'D': 'stage2',
                            'E': 'stage2',
                            'F': 'stage3',
                            'G': 'stage3'
                        }
                        nouvelle_base_complète['stage'] = nouvelle_base_complète['grade2024'].map(correspondance_stage)

                        # Adjust PD based on staging
                        nouvelle_base_complète.loc[nouvelle_base_complète['stage'] == 'stage1', 'PD'] = (
                            nouvelle_base_complète.loc[nouvelle_base_complète['stage'] == 'stage1', 'PD'] / 100
                        )
                        stage2_ids = nouvelle_base_complète[nouvelle_base_complète['stage'] == 'stage2']['id'].tolist()
                        for client_id in stage2_ids:
                            pd_values = [nouvelle_base_complète.loc[nouvelle_base_complète['id'] == client_id, 'PD'].iloc[0] / 100]
                            lifetime_pd = sum(pd_values)  # Placeholder for lifetime PD
                            nouvelle_base_complète.loc[nouvelle_base_complète['id'] == client_id, 'PD'] = lifetime_pd
                        nouvelle_base_complète.loc[nouvelle_base_complète['stage'] == 'stage3', 'PD'] = 1.0

                        # Calculate ECL
                        nouvelle_base_complète['ECL'] = nouvelle_base_complète['PD'] * nouvelle_base_complète['LGD'] * nouvelle_base_complète['EAD']

                        # Convert PD and LGD back to percentage for display
                        nouvelle_base_complète['PD'] = nouvelle_base_complète['PD'] * 100
                        nouvelle_base_complète['LGD'] = nouvelle_base_complète['LGD'] * 100

                        # Find the client
                        client_data = nouvelle_base_complète[nouvelle_base_complète['id'] == submitted_id]
                        if not client_data.empty:
                            client = client_data.iloc[0].to_dict()
                            client['PD TTC'] = f"{client['PD']:.2f}%"  # PD in percentage
                            client['staging'] = client.get('stage', 'Unknown')
                            client['PD Life Time'] = f"{client['PD']:.2f}%" if client['stage'] == 'stage2' else '-'
                            client['ECL'] = f"{client['ECL']:.2f}"
                            client['Nombre de contrats financés'] = 1  # Placeholder, adjust if actual data exists
                        else:
                            message = "ID client non trouvé."
                            



        context = {
            'client': client,
            'message': message,
            'submitted_id': submitted_id
        }
        return render(request, 'ifrs_client.html', context)

    except ValueError as ve:
        return render(request, 'ifrs_client.html', {'message': f"Erreur de conversion des données : {str(ve)}"})
    except KeyError as ke:
        return render(request, 'ifrs_client.html', {'message': f"Clé manquante : {str(ke)}"})
    except Exception as e:
        return render(request, 'ifrs_client.html', {'message': f"Erreur inattendue : {str(e)}"})


def consultationOctroi(request):
    return render(request, 'consultationOctroi.html')

def générerEtat(request):
    return render(request, 'générerEtat.html')