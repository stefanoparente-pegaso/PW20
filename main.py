import os.path
import pathlib
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.DatasetGenerator import generateDataset
from src.dataset_utils import preprocess_dataset, tokenize_text, embed_dataset
from src.train_models import train_model

# TODO: gestire path come variabili globali o init config??

dataset_path = "data/dataset.csv"
dataset_bck_path = "data/bck"
training_rows_percentage = 80
dep_model_path = "models/department_model.pkl"
sent_model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

# def init_config():
#     root_path = pathlib.Path(__file__).parent.resolve()
#     dataset_path = str(root_path / "data" / "dataset.csv")
#     dataset_bck_path = str(root_path / "data" / "bck")
#     dep_model_path = str(root_path / "models" / "department_model.pkl")
#     sent_model_path = str(root_path / "models" / "sentiment_model.pkl")
#     vectorizer_path = str(root_path / "models" / "vectorizer.pkl")
#     return dataset_path, dataset_bck_path, dep_model_path, sent_model_path, vectorizer_path


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


def view_preprocessed_dataset(dataset_path):
    dataframe = preprocess_dataset(dataset_path, training_rows_percentage, True)
    output_path = Path(dataset_path).parent / "preprocessed_dataset.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        for index, row in dataframe.iterrows():
            riga = f"{row['ID']} - {row['recensione_completa']} - {row['Reparto']} - {row['Sentiment']}\n"
            file.write(riga)
        print(f"Salvataggio completato in: {output_path}")


def train(dataset_path):
    dataframe = preprocess_dataset(dataset_path, training_rows_percentage, True)
    tokens = dataframe['recensione_completa'].apply(tokenize_text)
    vectorizer = TfidfVectorizer()
    rev_vector = embed_dataset(tokens, vectorizer)
    model_dep = train_model(rev_vector, dataframe['Reparto'])
    model_sent = train_model(rev_vector, dataframe['Sentiment'])
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model_dep, dep_model_path)
    joblib.dump(model_sent, sent_model_path)

def check_results(dataset_path):
    if not os.path.exists(dep_model_path) or not os.path.exists(sent_model_path):
        print("I modelli non sono ancora stati addestrati. Addestrare i modelli prima di lanciare questo comando.")
        return # TODO: return None ??


    return 1

def main():
    # dataset_path, dataset_bck_path = init_config()

    print("Benvenuto nel programma di machine learning PW20.")
    scelta = ""
    while True:
        print_menu()
        scelta = input("Scegli la funzionalità da eseguire: ")
        match scelta:
            case "0": return
            case "1": generateDataset(dataset_path, dataset_bck_path)
            case "2": view_preprocessed_dataset(dataset_path)
            case "3": train(dataset_path)
            case "4": check_results(dataset_path)
        print()
        print("====================================================")


if __name__ == "__main__":
    main()
