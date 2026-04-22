Versione Python usata per creare progetto: 3.14.3

1. Copiarsi il progetto in locale eseguendo il seguente comando oppure scaricandolo come zip

```commandline
git clone https://github.com/stefanoparente-pegaso/PW20.git
```

2. Entrare nella cartella `PW20` e creare un virtual environment con il seguente comando

```commandline
python -m venv venv
```

Poi attivarla con

```commandline
venv\Scripts\activate
```

3. Scaricare le dipendenze di progetto con il seguente comando

```commandline
pip install -r requirements.txt
```

4. Avvia il programma con il seguente comando

```commandline
python main.py
```

Si aprirà un menu interattivo da terminale in cui verranno indicate le operazioni eseguibili:

```text
0. Esci dal programma
1. Genera un nuovo dataset (prototipo non funzionante)
2. Crea un .txt che mostra dataset pre-processato
3. Addestra i modelli di ML
4. Visualizza risultati dei modelli di ML sul dataset fornito con una dashboard statica all'interno del browser
5. Apri l'interfaccia' interattiva in cui inserire recensioni a mano ed esportarle
```

Le operazioni 4 e 5 eseguono automaticamente l'operazione 3 qualora i modelli non fossero ancora stati addestrati.

In caso si scelga 4 si aprirà sul browser una dashboard statica in html


#### Info interfaccia Gradio (5)

In caso si scelga 5 si aprirà l'interfaccia Gradio su cui è possibile inserire recensioni, valutarle, esportarle in CSV o importarle da file CSV esistente e valido.

- Per aggiornare la tabella dopo aver inserito una o più recensioni bisogna espanderla e ricomprimerla
- Per uscire dall'interfaccia ed eseguire un'altra funzionalità tornare sul terminale e premere `CTRL+C` una volta