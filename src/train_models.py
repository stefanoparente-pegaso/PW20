from sklearn.linear_model import LogisticRegression

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