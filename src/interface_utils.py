import gradio as gr
import pandas as pd
from datetime import datetime
from src.dataset_utils import clean_text, tokenize_text


def predict(titolo, corpo, v_dep, v_sent, model_dep, model_sent):
    recensione = titolo + ' ' + corpo
    recensione_pulita = clean_text(recensione)

    tokens_dep_list = tokenize_text(recensione_pulita, sentiment=False)
    print(f"DEBUG REPARTO TOKENS (Nomi): {tokens_dep_list}")  # Debug per i nomi/oggetti

    tokens_dep_str = " ".join(tokens_dep_list)
    vettore_dep = v_dep.transform([tokens_dep_str])
    dep = model_dep.predict(vettore_dep)[0]

    tokens_sent_list = tokenize_text(recensione_pulita, sentiment=True)
    print(f"DEBUG SENTIMENT TOKENS (Adj/Adv): {tokens_sent_list}")  # Debug per gli aggettivi

    tokens_sent_str = " ".join(tokens_sent_list)
    vettore_sent = v_sent.transform([tokens_sent_str])
    sent = model_sent.predict(vettore_sent)[0]

    score = float(max(model_sent.predict_proba(vettore_sent)[0]))

    return dep, sent, score


def launch_gradio(v_dep, v_sent, model_dep, model_sent):
    with gr.Blocks(title="Classificatore recensioni hotel") as demo:
        gr.Markdown("# Classificatore recensioni hotel")

        def import_csv(file, current_data):
            if file is None:
                return current_data
            try:
                df_imported = pd.read_csv(file.name, sep=';')
                if "Titolo" in df_imported.columns and "Corpo" in df_imported.columns:
                    df_to_process = df_imported[["Titolo", "Corpo"]].dropna()
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
                    updated_data = pd.concat([current_data, pd.DataFrame(results)], ignore_index=True)
                    return updated_data
                else:
                    gr.Warning("Il file CSV deve contenere 'Titolo' e 'Corpo'.")
                    return current_data
            except Exception as e:
                gr.Error(f"Errore: {str(e)}")
                return current_data

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
        rev_table = gr.Dataframe(
            headers=["Titolo", "Corpo", "Reparto", "Sentiment"],
            datatype=["str", "str", "str", "str"],
            value=pd.DataFrame(columns=["Titolo", "Corpo", "Reparto", "Sentiment"]),
            interactive=False
        )

        with gr.Row():
            btn_export = gr.Button("Esporta in CSV")
            file_download = gr.File(label="Scarica file")

        btn_run.click(fn=analyze, inputs=[txt_titolo, txt_corpo, rev_table], outputs=[out_dep, out_sent, out_score, rev_table])
        btn_import.click(fn=import_csv, inputs=[file_upload, rev_table], outputs=[rev_table])
        btn_export.click(fn=lambda d: d.to_csv("export.csv", index=False, sep=';') or "export.csv", inputs=[rev_table], outputs=[file_download])

    demo.launch(inbrowser=True)