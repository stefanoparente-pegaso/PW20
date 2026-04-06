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
vectorizer_path = str(models_dir / "vectorizer.pkl")
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
    tokens = dataframe_80['recensione_completa'].apply(tokenize_text)
    vectorizer = TfidfVectorizer()
    rev_vector = embed_dataset(tokens, vectorizer)
    model_dep = train_model(rev_vector, dataframe_80['Reparto'])
    model_sent = train_model(rev_vector, dataframe_80['Sentiment'])
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model_dep, dep_model_path)
    joblib.dump(model_sent, sent_model_path)

def check_results():
    if not os.path.exists(dep_model_path) or not os.path.exists(sent_model_path):
        print("I modelli non sono ancora stati addestrati. Verrà eseguito anche addestramento")
        train()
    models = []
    dataframe_20 = preprocess_dataset(dataset_path, 100 - training_rows_percentage, False)
    tokens_20 = dataframe_20['recensione_completa'].apply(tokenize_text)
    strings_20 = tokens_20.apply(lambda x: " ".join(x))
    vectorizer = joblib.load(vectorizer_path)
    rev_vector = vectorizer.transform(strings_20)
    model_dep = joblib.load(dep_model_path)
    model_sent = joblib.load(sent_model_path)
    model_result_dep = evaluate_model('DEPARTMENT MODEL', model_dep, rev_vector, dataframe_20['Reparto'], dep_matrix_labels, dataframe_20['ID'])
    model_result_sent = evaluate_model('SENTIMENT MODEL', model_sent, rev_vector, dataframe_20['Sentiment'], sent_matrix_labels, dataframe_20['ID'])
    models.append(model_result_dep)
    models.append(model_result_sent)
    show_results(models)

def open_interface():
    if not os.path.exists(dep_model_path) or not os.path.exists(sent_model_path):
        print("I modelli non sono ancora stati addestrati. Verrà eseguito anche addestramento")
        train()
    vectorizer = joblib.load(vectorizer_path)
    model_dep = joblib.load(dep_model_path)
    model_sent = joblib.load(sent_model_path)
    launch_gradio(vectorizer, model_dep, model_sent)

    return ""
