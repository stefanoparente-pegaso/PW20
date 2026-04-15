from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def train_model(vector, column_df):
    lgr = LogisticRegression(
        random_state=42,
        max_iter=2000,
        class_weight='balanced',
        C=1.0,
        solver='lbfgs',
        #multi_class='auto'
    )

    # le categorie della colonna vengono automaticamente convertite in ordine alfabetico
    model = lgr.fit(vector, column_df)
    return model


# def train_model(vector, column_df):
#     # Basta cambiare la classe del modello
#     model = LinearSVC(
#         random_state=42,
#         max_iter=2000,
#         class_weight='balanced', # Fondamentale per i tuoi reparti sbilanciati
#         C=1.0                    # Parametro di regolarizzazione (puoi testare 0.1 o 1.0)
#     )
#
#     model.fit(vector, column_df)
#     return model