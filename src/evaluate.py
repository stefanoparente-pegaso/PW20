import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def show_results(accuracy, f1, conf_matrix):
    return 2

def evaluate_model(model, rev_vector, column_df):

    prediction = model.predict(rev_vector)
    accuracy = accuracy_score(column_df, prediction)
    f1 = f1_score(column_df, prediction)
    conf_matrix = confusion_matrix(column_df, prediction)

    return accuracy, f1, conf_matrix