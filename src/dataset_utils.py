import re

import pandas as pd
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nlp = spacy.load("it_core_news_sm")

def embed_dataset(rev_tk_df, vectorizer):
    rev_strings = rev_tk_df.apply(lambda x: " ".join(x))
    rev_tfidf = vectorizer.fit_transform(rev_strings)
    return rev_tfidf


# def tokenize_text(text):
#     # Processiamo il testo con spaCy
#     doc = nlp(text)
#
#     tokens = []
#     for token in doc:
#         # Teniamo il "non" (fondamentale per il sentiment!)
#         # e filtriamo stopword, punteggiatura e spazi
#         if token.text.lower() == "non":
#             tokens.append("non")
#         elif not token.is_stop and not token.is_punct and not token.is_space:
#             # Prendiamo il lemma (la radice della parola)
#             tokens.append(token.lemma_.lower())
#
#     return tokens

def tokenize_text(rev_df):
    tokens = word_tokenize(rev_df) # lista
    stop_words = set(stopwords.words('italian'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


# def clean_text(text):
#     if not isinstance(text, str):
#         return ""
#
#     text = text.lower().strip()
#
#     # Sostituiamo ogni carattere di punteggiatura con uno spazio
#     # [^\w\s] significa: "qualsiasi cosa che non sia una lettera o uno spazio"
#     text = re.sub(r'[^\w\s]', ' ', text)
#
#     # Riduciamo i doppi spazi che si creano in uno spazio singolo
#     text = re.sub(r'\s+', ' ', text).strip()
#
#     return text

def clean_text(text):
    text = text.lower().strip().translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_dataset(dataset_path, rows_to_return_percentage, ascending):
    # nltk.download('punkt') --> se si rompe riattivare
    df = pd.read_csv(dataset_path, sep=';')
    df['recensione_completa'] = df['Titolo'] + " " + df['Corpo']
    df['recensione_completa'] = df['recensione_completa'].apply(clean_text)

    total_rows = len(df)
    rows_to_return = total_rows * rows_to_return_percentage // 100
    if ascending:
        return df.head(rows_to_return)
    else:
        return df.tail(rows_to_return)

