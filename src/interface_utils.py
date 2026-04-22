import os
import tempfile

import gradio as gr
import pandas as pd
from datetime import datetime
from src.dataset_utils import clean_text, tokenize_text


def predict(titolo, corpo, v_dep, v_sent, model_dep, model_sent):

    recensione = clean_text(titolo + ' ' + corpo)

    # tokenizzazione, vettorizzazione e predizione department
    tokens_dep_list = tokenize_text(recensione, sentiment=False)
    tokens_dep_str = " ".join(tokens_dep_list)
    vettore_dep = v_dep.transform([tokens_dep_str])
    dep = model_dep.predict(vettore_dep)[0]

    # tokenizzazione, vettorizzazione e predizione con score sentiment
    tokens_sent_list = tokenize_text(recensione, sentiment=True)
    tokens_sent_str = " ".join(tokens_sent_list)
    vettore_sent = v_sent.transform([tokens_sent_str])
    sent = model_sent.predict(vettore_sent)[0]

    score = float(max(model_sent.predict_proba(vettore_sent)[0]))

    return dep, sent, score


def launch_gradio(v_dep, v_sent, model_dep, model_sent):
    # Definizione pagina (block) "prediction_interface"
    with gr.Blocks(title="Classificatore recensioni hotel") as prediction_interface:
        gr.Markdown("# Classificatore recensioni hotel")

        # Unisco record file input con quelli già inseriti in tabella
        def import_csv(file, current_data):
            if file is None:
                return current_data
            try:
                df_imported = pd.read_csv(file.name, sep=';') # Lettura file
                if "Titolo" in df_imported.columns and "Corpo" in df_imported.columns: # Validazione colonne
                    df_to_process = df_imported[["Titolo", "Corpo"]].dropna() # Rimozione tutti record con uno dei campi vuoto

                    # Analisi delle recensioni del file e aggiunta
                    results = []
                    for _, row in df_to_process.iterrows():
                        dep, sent, score = predict(
                            row["Titolo"], row["Corpo"],
                            v_dep, v_sent, model_dep, model_sent
                        )
                        results.append({
                            "Titolo": row["Titolo"],
                            "Corpo": row["Corpo"],
                            "Reparto": dep,
                            "Sentiment": sent
                        })
                    # Concatenazione tabella presente e recensioni file; ignore_index per ricalcolo indici per evitare duplicati
                    updated_data = pd.concat([current_data, pd.DataFrame(results)], ignore_index=True)
                    return updated_data
                else:
                    gr.Warning("File CSV non valido: deve contenere colonne 'Titolo' e 'Corpo'.")
                    return current_data
            except Exception as e:
                gr.Error(f"Errore: {str(e)}")
                return current_data

        # Inserimento nuova recensione nella tabella già presente
        def analyze(titolo, corpo, rev_list):
            dep, sent, score = predict(titolo, corpo, v_dep, v_sent, model_dep, model_sent)
            new_entry = {
                "Titolo": titolo,
                "Corpo": corpo,
                "Reparto": dep,
                "Sentiment": sent,
            }
            updated_data = pd.concat([rev_list, pd.DataFrame([new_entry])], ignore_index=True)
            return dep, sent, score, updated_data

        # Si salva tra i file temporanei per non intasare la root del progetto
        def export (df):
            name = f"recensioni_esportate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            temp_dir = tempfile.gettempdir()
            path = os.path.join(temp_dir, name)
            df.to_csv(path, index=False, sep=';')
            return path

        # Definizione grafica prima riga della pagina
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Importa recensioni CSV")
                file_upload = gr.File(label="Carica CSV", file_types=[".csv"])
                btn_import = gr.Button("Analizza file caricato")

            with gr.Column():
                gr.Markdown("#### Inserisci recensione")
                txt_titolo = gr.Textbox(label="Titolo")
                txt_corpo = gr.Textbox(label="Testo", lines=4)
                btn_run = gr.Button("Analizza", variant="primary")

            with gr.Column():
                out_dep = gr.Label(label="Reparto Destinatario")
                out_sent = gr.Label(label="Sentiment Rilevato")
                out_score = gr.Number(label="Confidenza (0-1)")

        gr.Markdown("---")

        # Definizione tabella sulla seconda riga
        rev_table = gr.Dataframe(
            headers=["Titolo", "Corpo", "Reparto", "Sentiment"],
            datatype=["str", "str", "str", "str"],
            value=pd.DataFrame(columns=["Titolo", "Corpo", "Reparto", "Sentiment"]),
            interactive=False # Blocco modifica
        )

        # Terza riga pagina con funzioni di export
        with gr.Row():
            btn_export = gr.Button("Esporta in CSV")
            file_download = gr.File(label="Scarica file")

        # Definizione pulsanti
        btn_run.click(fn=analyze, inputs=[txt_titolo, txt_corpo, rev_table], outputs=[out_dep, out_sent, out_score, rev_table])
        btn_import.click(fn=import_csv, inputs=[file_upload, rev_table], outputs=[rev_table])
        btn_export.click(fn=export, inputs=[rev_table], outputs=[file_download])

    prediction_interface.launch(inbrowser=True)