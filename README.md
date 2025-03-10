# [Metodi Numerici per Equazioni alle Derivate Parziali](https://luca-heltai.github.io/nmpde/)

[![deploy-book](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml/badge.svg)](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml)

## L'ambiente jupyter-book

Per generare le pagine web con jupyter-book è conveniente creare un *virtual environment* ed
utilizzare il file `requirements.txt` distribuito nel repository:

1. `python3 -m pip install --user virtualenv` (Installare virtualenv),
2. `python3 -m venv jupyterbook` (Creare un virtual environment),
3. `source jupyterbook/bin/activate` (Attivare il virtual environment),
4. `python3 -m pip install -r requirements.txt` (Installare i pacchetti necessari).

Tutte le volte che si vorrà aggiornare il materiale relativo ai laboratori sarà
sufficiente avviare il *virtual environment*:

```
source jupyterbook/bin/activate
```

navigare nella cartella del repository ed utilizzare `make`:

- `make clean` ripulisce la distribuzione,
- `make build` costruisce le pagine html,
- `make show` costruisce le pagine e le mostra in un browser locale,
- `make publish` utilizza `gh-pages` per pubblicare le pagine web,
- `make latex` produce una versione `.tex` dei laboratori.

Lo script `start.sh` esegue per te l'inizializzazione del *virtual environment* `jupyterbook`. Se non hai mai inizializzato il *virtual environment*, ne crea uno per te (eseguendo esattamente i comandi sopra indicati) e lo carica, altrimenti si limita a caricare il *virtual environment* esistente.

Ad ogni commit su `main`, le pagine all'indirizzo <https://luca-heltai.github.io/nmpde/> vengono rigenerate in modo automatico usando la *github action* qui sotto.

[![deploy-book](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml/badge.svg)](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml)
