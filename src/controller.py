import os.path
import pathlib
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.DatasetGenerator import generateDataset
from src.dataset_utils import preprocess_dataset, tokenize_text, embed_dataset
from src.train_models import train_model
from src.evaluate import evaluate_model, show_results
from src.interface_utils import launch_gradio


# TODO: valutare di spostare funzioni train, show, results in un file dedicato per mantenere pulito il main con solo opzioni menu

# Definizione root e creazione cartelle se non presenti
root_path = pathlib.Path(__file__).parent.parent.resolve()
data_dir = root_path / "data"
models_dir = root_path / "trained_models"
data_dir.mkdir(exist_ok=True)
models_dir.mkdir(exist_ok=True)

# Definizioni costanti path e righe da processare
dataset_path = str(data_dir / "dataset.csv")
dataset_bck_path = str(data_dir / "bck")
dep_model_path = str(models_dir / "department_model.pkl")
sent_model_path = str(models_dir / "sentiment_model.pkl")
vectorizer_sent_path = str(models_dir / "vectorizer_sent.pkl")
vectorizer_dep_path = str(models_dir / "vectorizer_dep.pkl")
training_rows_percentage = 80

# In ordine alfabetico
dep_matrix_labels = ['F&B', 'Housekeeping', 'Reception']
sent_matrix_labels = ['Negativo', 'Positivo']

def generate_dataset():
    generateDataset(dataset_path, dataset_bck_path)

def print_menu():
    print()
    print("Le funzionalità disponibili sono le seguenti:")
    print("0. Esci dal programma")
    print("1. Genera un nuovo dataset")
    print("2. Visualizza il dataset corrente dopo l'operazione di preprocessing")
    print("3. Addestra i modelli di ML")
    print("4. Visualizza risultati dei modelli di ML sul dataset fornito")
    print("5. Apri la dashboard interattiva")
    print()


def view_preprocessed_dataset():
    dataframe = preprocess_dataset(dataset_path, training_rows_percentage, True)
    output_path = Path(dataset_path).parent / "preprocessed_dataset.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for index, row in dataframe.iterrows():
            riga = f"{row['ID']} - {row['recensione_completa']} - {row['Reparto']} - {row['Sentiment']}\n"
            file.write(riga)
        print(f"Salvataggio completato in: {output_path}")


def train():
    dataframe_80 = preprocess_dataset(dataset_path, training_rows_percentage, True)

    tokens_dep = dataframe_80['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=False))
    vectorizer_dep = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, use_idf=True) # TODO: provare use_idf = True
    # Trasformiamo i token in stringhe per il vectorizer
    rev_strings_dep = tokens_dep.apply(lambda x: " ".join(x))
    rev_vector_dep = vectorizer_dep.fit_transform(rev_strings_dep)

    model_dep = train_model(rev_vector_dep, dataframe_80['Reparto'])

    tokens_sent = dataframe_80['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=True))
    vectorizer_sent = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.7, sublinear_tf=True, use_idf=False) # TODO: provare min = 1 e use_idf = True
    rev_strings_sent = tokens_sent.apply(lambda x: " ".join(x))
    rev_vector_sent = vectorizer_sent.fit_transform(rev_strings_sent)

    model_sent = train_model(rev_vector_sent, dataframe_80['Sentiment'])

    import numpy as np
    feature_names = vectorizer_sent.get_feature_names_out()
    print("\n--- ANALISI DEI PESI DEL MODELLO SENTIMENT (Solo ADJ/ADV) ---")
    for i, class_label in enumerate(model_sent.classes_):
        if len(model_sent.classes_) <= 2:
            weights = model_sent.coef_[0] if i == 1 else -model_sent.coef_[0]
        else:
            weights = model_sent.coef_[i]
        top10_idx = np.argsort(weights)[-20:]
        print(f"Top 10 parole per {class_label}: {[feature_names[j] for j in top10_idx]}")
    print("------------------------------------\n")

    joblib.dump(vectorizer_dep, vectorizer_dep_path)
    joblib.dump(vectorizer_sent, vectorizer_sent_path)
    joblib.dump(model_dep, dep_model_path)
    joblib.dump(model_sent, sent_model_path)


def check_results():
    if not os.path.exists(dep_model_path) or not os.path.exists(sent_model_path):
        print("I modelli non sono ancora stati addestrati. Verrà eseguito anche addestramento")
        train()

    models = []
    dataframe_20 = preprocess_dataset(dataset_path, 100 - training_rows_percentage, False)

    tokens_dep = dataframe_20['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=False))
    strings_dep = tokens_dep.apply(lambda x: " ".join(x))
    vectorizer_dep = joblib.load(vectorizer_dep_path)
    rev_vector_dep = vectorizer_dep.transform(strings_dep)

    tokens_sent = dataframe_20['recensione_completa'].apply(lambda x: tokenize_text(x, sentiment=True))
    strings_sent = tokens_sent.apply(lambda x: " ".join(x))
    vectorizer_sent = joblib.load(vectorizer_sent_path)
    rev_vector_sent = vectorizer_sent.transform(strings_sent)

    # Caricamento modelli
    model_dep = joblib.load(dep_model_path)
    model_sent = joblib.load(sent_model_path)

    # model_result_dep = evaluate_model('DEPARTMENT MODEL', model_dep, rev_vector_dep, dataframe_20['Reparto'], dep_matrix_labels, dataframe_20['ID'])
    model_result_dep = evaluate_model('DEPARTMENT MODEL', model_dep, rev_vector_dep, dataframe_20['Reparto'], dep_matrix_labels, dataframe_20['ID'], dataframe_20['recensione_completa'])
    # model_result_sent = evaluate_model('SENTIMENT MODEL', model_sent, rev_vector_sent, dataframe_20['Sentiment'], sent_matrix_labels, dataframe_20['ID'])
    model_result_sent = evaluate_model('SENTIMENT MODEL', model_sent, rev_vector_sent, dataframe_20['Sentiment'], sent_matrix_labels, dataframe_20['ID'], dataframe_20['recensione_completa'])

    models.append(model_result_dep)
    models.append(model_result_sent)
    show_results(models)


def open_interface():
    if not os.path.exists(dep_model_path) or not os.path.exists(sent_model_path):
        print("I modelli non sono ancora stati addestrati. Verrà eseguito anche addestramento")
        train()

    v_dep = joblib.load(vectorizer_dep_path)
    v_sent = joblib.load(vectorizer_sent_path)

    # Carichiamo i modelli
    model_dep = joblib.load(dep_model_path)
    model_sent = joblib.load(sent_model_path)

    # Passiamo tutto alla funzione launch_gradio
    # NOTA: dovrai aggiornare la funzione launch_gradio in interface_utils.py
    # affinché accetti questi 4 parametri invece di 3.
    launch_gradio(v_dep, v_sent, model_dep, model_sent)

    return ""
