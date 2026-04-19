import os
import webbrowser

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from models.model_result import ModelResult
from .dashboard_html_utils import get_html_model
import pandas as pd

def show_results(models_to_show):
    dashboard_dir = './static/evaluation_dashboard/'
    template_path = os.path.join(dashboard_dir, 'dashboard.html')
    output_path = os.path.join(dashboard_dir, 'report_generato.html')

    try:

        html_to_replace_list = []

        for model in models_to_show:
            results_html = get_html_model(model)
            html_to_replace_list.append(results_html)

        html_to_replace = ' '.join(html_to_replace_list)
        content = ''

        with open(template_path, 'r') as f:
            content = f.read()

        dashboard = content.replace('%%CARDS_PLACEHOLDER%%', html_to_replace)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard)

        webbrowser.open(f"file://{os.path.abspath(output_path)}")

    except FileNotFoundError:
        print("Errore nel caricamento della dashboard, si mostrano i risultati a terminale:")
        for model in models_to_show:
            print(model.to_dict())

def get_errors_csv(model_name, prediction, column_df, id_df):
    results = pd.DataFrame({
        'ID': id_df,
        'Reale': column_df,
        'Predetto': prediction
    })

    errors = results[results['Reale'] != results['Predetto']]
    target_dir = os.path.join('data', 'prediction_errors')
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.join(target_dir, f"errori_{model_name}.csv")
    errors.to_csv(filename, index=False, sep=';', encoding='utf-8')

# DEBUG

def evaluate_model(name, model, rev_vector, column_df, labels, id_df, original_texts):
    prediction = model.predict(rev_vector)
    accuracy = accuracy_score(column_df, prediction)
    f1 = f1_score(column_df, prediction, average='macro')
    conf_matrix = confusion_matrix(column_df, prediction)

    # --- DEBUG ERRORI SPECIFICI ---
    print(f"\n--- ANALISI ERRORI TARGET: {name} ---")

    # Creiamo un piccolo df temporaneo per l'analisi
    df_debug = pd.DataFrame({
        'Testo': original_texts,
        'Reale': column_df.values,
        'Predetto': prediction
    })

    # Filtro 1: Era Reception ma ha predetto altro
    if name == 'DEPARTMENT MODEL':
        errori_reception = df_debug[(df_debug['Reale'] == 'Reception') & (df_debug['Predetto'] != 'Reception')]
        print(f"\nERRORE: Erano RECEPTION ma il modello ha detto altro ({len(errori_reception)} casi):")
        for i, row in errori_reception.head(10).iterrows():  # Vediamo i primi 10
            print(f"- SCRITTO: {row['Testo'][:80]}... | PREDETTO: {row['Predetto']}")

    # Filtro 2: Era Negativo ma ha predetto Positivo
    if name == 'SENTIMENT MODEL':
        falsi_positivi = df_debug[(df_debug['Reale'] == 'Negativo') & (df_debug['Predetto'] == 'Positivo')]
        print(f"\nERRORE: Erano NEGATIVI ma il modello ha detto POSITIVO ({len(falsi_positivi)} casi):")
        for i, row in falsi_positivi.head(10).iterrows():
            print(f"- SCRITTO: {row['Testo'][:80]}... | PREDETTO: {row['Predetto']}")

    print("---------------------------------------\n")
    # ------------------------------

    result = ModelResult(name, accuracy, f1, conf_matrix, labels)
    get_errors_csv(name, prediction, column_df, id_df)

    return result
