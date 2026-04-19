import pandas as pd
import string
import spacy
# import nltk

# Algoritmi di tokenizzazione nltk utilizzati nella prima versione

# nltk.download('punkt', quiet=True) # quiet=True serve per non far stampare ogni volta i log
# nltk.download('punkt_tab', quiet=True)
# nltk.download('stopwords', quiet=True)

# Modello linguistico
nlp = spacy.load('it_core_news_md') # Il modello small sbarella troppo, quello medio ok

def embed_dataset(rev_tk_df, vectorizer):
    rev_strings = rev_tk_df.apply(lambda x: ' '.join(x))
    rev_tfidf = vectorizer.fit_transform(rev_strings)
    return rev_tfidf


def tokenize_text(text, sentiment):
    doc = nlp(text)
    tokens = []

    # Per il modello sentiment considero solo aggettivi, avverbi e verbi. Per department anche nomi e pronomi
    word_types_sentiment = ['ADJ', 'ADV', 'VERB']
    word_types_department = ['NOUN', 'PROPN', 'VERB', 'ADJ']

    for token in doc:
        # Ogni parola viene 'lemmizzata' ovvero messa in forma base (es. buoni --> buono)
        lemma = token.lemma_.lower()

        if sentiment:
            # Si evita che il termine 'non' venga tolto dal nlp, ci serve per negazione aggettivi positivi
            if lemma == 'non':
                tokens.append('non')
            # Si filtrano solo le parti richieste
            elif token.pos_ in word_types_sentiment:
                tokens.append(lemma)
        else:
            if token.pos_ in word_types_department:
                # Non si considerano caratteri inutili se presenti
                if not token.is_stop and not token.is_punct and not token.is_space:
                    tokens.append(lemma)

    return tokens

def clean_text(text):
    # Rimozione punteggiatura, spazi e tutto in lowercase
    text = text.lower().strip().translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_dataset(dataset_path, rows_to_return_percentage, ascending):
    # Si trasforma il dataset CSV in un dataframe pandas
    df = pd.read_csv(dataset_path, sep=';')

    # df['recensione_completa'] = df['Titolo'] + ' ' + df['Titolo'] + ' ' + df['Corpo'] # Per dare più peso al titolo
    df['recensione_completa'] = df['Titolo'] + ' ' + df['Corpo']
    df['recensione_completa'] = df['recensione_completa'].apply(clean_text)

    # Settaggio della percentuale di record da restituire e se partire di cima o da fondo
    # In questo caso nel controller passeremo per training 80 da cima, per evaluation 20 da fondo
    total_rows = len(df)
    rows_to_return = total_rows * rows_to_return_percentage // 100
    if ascending:
        return df.head(rows_to_return)
    else:
        return df.tail(rows_to_return)

