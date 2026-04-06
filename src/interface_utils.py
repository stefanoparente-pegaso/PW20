import gradio as gr
from src.dataset_utils import clean_text, tokenize_text, embed_dataset

def launch_gradio(vectorizer, model_dep, model_sent):

    def predict(titolo, corpo):
        recensione = titolo + ' ' + corpo
        recensione = clean_text(recensione)
        tokens = " ".join(tokenize_text(recensione))
        vettore = vectorizer.transform([tokens])
        dep = model_dep.predict(vettore)[0]
        sent = model_sent.predict(vettore)[0]
        # predict_proba restituisce entrambi score x positivo e negativo
        score = float(max(model_sent.predict_proba(vettore)[0]))
        return dep, sent, score

    with gr.Blocks(title="Classificatore recensioni hotel") as demo:
        gr.Markdown("# Classificatore recensioni hotel")
        gr.Markdown("Inserisci il titolo e il corpo della recensione:")

        with gr.Row():
            with gr.Column():
                txt_titolo = gr.Textbox(label="Titolo", placeholder="Inserisci qui il titolo")
                txt_corpo = gr.Textbox(label="Testo", placeholder="Inserisci qui la recensione", lines=4)
                btn_run = gr.Button("Analizza Recensione", variant="primary")

            with gr.Column():
                out_dep = gr.Label(label="Reparto Destinatario")
                out_sent = gr.Label(label="Sentiment Rilevato")
                # Usiamo un componente Label o Number per la probabilità richiesta dal PW
                out_score = gr.Number(label="Confidenza Modello (0-1)")

        btn_run.click(
            fn=predict,
            inputs=[txt_titolo, txt_corpo],
            outputs=[out_dep, out_sent, out_score]
        )

    demo.launch(inbrowser=True)

    return
