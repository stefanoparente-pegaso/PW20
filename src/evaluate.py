import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from models.model_result import ModelResult


def show_results(models_to_show):

    for model in models_to_show:
        print('test')

    return 2

def evaluate_model(name, model, rev_vector, column_df):

    prediction = model.predict(rev_vector)
    accuracy = accuracy_score(column_df, prediction)
    f1 = f1_score(column_df, prediction, average='macro') #TODO: questo è ok per entrambi i modelli?
    conf_matrix = confusion_matrix(column_df, prediction)

    result = ModelResult(name, accuracy, f1, conf_matrix)

    return result