from sklearn.linear_model import LogisticRegression

def train_model(vector, column_df):
    lgr = LogisticRegression()
    model = lgr.fit(vector, column_df)
    return model