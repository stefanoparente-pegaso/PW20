import os
import webbrowser

import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from models.model_result import ModelResult
from .dashboard_html_utils import get_html_model
import pandas as pd

from .dataset_utils import tokenize_text

# In ordine alfabetico
dep_matrix_labels = ['F&B', 'Housekeeping', 'Reception']
sent_matrix_labels = ['Negativo', 'Positivo']


def show_results(models_to_show, errors_html):
    # Path dashboard
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

        # Apro template, leggo e metto in content, sostituisco placeholder, salvo file, lo apro
        with open(template_path, 'r') as f:
            content = f.read()

        dashboard = content.replace('%%CARDS_PLACEHOLDER%%', html_to_replace)
        dashboard = dashboard.replace('%%ERRORS_PLACEHOLDER%%', errors_html)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard)

        webbrowser.open(f"file://{os.path.abspath(output_path)}") # Apertura dashboard browser predefinito

    except FileNotFoundError:
        print("Errore nel caricamento della dashboard, si mostrano i risultati a terminale:")
        for model in models_to_show:
            print(model.to_dict())


def get_errors_html(model_name, prediction, column_df, id_df, original_texts):
    # Dataframe temporaneo
    results = pd.DataFrame({
        'ID': id_df,
        'Testo': original_texts,
        'Reale': column_df,
        'Predetto': prediction
    })

    # Filtro solo record in cui la predizione non corrisponde al reale
    errors = results[results['Reale'] != results['Predetto']]

    if errors.empty:
        return f"<p class='no-errors'>Nessun errore rilevato per {model_name}.</p>"

    html_table = f"<h3 class='error-title'>Analisi errori: {model_name}</h3>"
    html_table += '<table class="error-table">'
    html_table += '<thead><tr>'
    html_table += '<th class="col-id">ID</th>'
    html_table += '<th class="col-text">Testo recensione completo</th>'
    html_table += '<th class="col-real">Reale</th>'
    html_table += '<th class="col-pred">Predetto</th>'
    html_table += '</tr></thead><tbody>'

    for _, row in errors.head(10).iterrows(): # _ è indice
        # Pulizia testo x visualizzazione
        testo_show = str(row['Testo']).replace('\n', ' ').strip()

        html_table += '<tr>'
        html_table += f'<td class="col-id">{row["ID"]}</td>'
        html_table += f'<td class="col-text">{testo_show}</td>'
        html_table += f'<td class="col-real">{row["Reale"]}</td>'
        html_table += f'<td class="col-pred">{row["Predetto"]}</td>'
        html_table += '</tr>'

    html_table += '</tbody></table>'
    return html_table

def evaluate_model(name, model, rev_vector, column_df, labels, id_df, original_texts):
    prediction = model.predict(rev_vector)
    accuracy = accuracy_score(column_df, prediction)
    f1 = f1_score(column_df, prediction, average='macro')
    conf_matrix = confusion_matrix(column_df, prediction)

    result = ModelResult(name, accuracy, f1, conf_matrix, labels)

    errors_html = get_errors_html(name, prediction, column_df, id_df, original_texts)

    return result, errors_html

def open_results_dashboard(dataframe_20, pkl_paths):

    models = []

    tokens_dep = dataframe_20['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=False))
    tokens_dep_str = tokens_dep.apply(lambda x: " ".join(x))
    vectorizer_dep = joblib.load(pkl_paths['vectorizer_dep_path'])
    rev_vector_dep = vectorizer_dep.transform(tokens_dep_str)

    tokens_sent = dataframe_20['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=True))
    tokens_sent_str = tokens_sent.apply(lambda x: " ".join(x))
    vectorizer_sent = joblib.load(pkl_paths['vectorizer_sent_path'])
    rev_vector_sent = vectorizer_sent.transform(tokens_sent_str)

    # Caricamento modelli
    model_dep = joblib.load(pkl_paths['dep_model_path'])
    model_sent = joblib.load(pkl_paths['sent_model_path'])

    # model_result_dep = evaluate_model('DEPARTMENT MODEL', model_dep, rev_vector_dep, dataframe_20['Reparto'], dep_matrix_labels, dataframe_20['ID'])
    model_result_dep, err_dep_html = evaluate_model('DEPARTMENT MODEL', model_dep, rev_vector_dep, dataframe_20['Reparto'],
                                      dep_matrix_labels, dataframe_20['ID'], dataframe_20['recensione_completa'])
    # model_result_sent = evaluate_model('SENTIMENT MODEL', model_sent, rev_vector_sent, dataframe_20['Sentiment'], sent_matrix_labels, dataframe_20['ID'])
    model_result_sent, err_sent_html = evaluate_model('SENTIMENT MODEL', model_sent, rev_vector_sent, dataframe_20['Sentiment'],
                                       sent_matrix_labels, dataframe_20['ID'], dataframe_20['recensione_completa'])

    errors_html = err_dep_html + "<br>" + err_sent_html

    models.append(model_result_dep)
    models.append(model_result_sent)
    show_results(models, errors_html)