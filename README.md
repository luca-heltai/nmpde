# [Metodi Numerici per Equazioni alle Derivate Parziali](https://luca-heltai.github.io/nmpde/)

[![deploy-book](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml/badge.svg)](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml)

## L'ambiente jupyter-book

Per generare le pagine web con jupyter-book si usa la virtualenv del repository e il file
`requirements.txt`. Sul computer del docente `start.sh` preferisce
`~/anaconda3/bin/python3`; sulle altre macchine usa il primo `python3` disponibile.

Per inizializzare e attivare l'ambiente nella shell corrente:

```
source ./start.sh
```

Lo script ricrea automaticamente una virtualenv mancante o corrotta. In alternativa
può eseguire direttamente un comando JupyterBook:

```
./start.sh build notes/
```

Dopo l'attivazione, navigare nella cartella del repository e utilizzare `make`:

- `make clean` ripulisce la distribuzione,
- `make build` costruisce le pagine html,
- `make show` costruisce le pagine e le mostra in un browser locale,
- `make publish` utilizza `gh-pages` per pubblicare le pagine web,
- `make latex` produce una versione `.tex` dei laboratori.

Il workflow GitHub Actions esegue gli stessi passaggi su Python 3.13 e pubblica automaticamente
il contenuto costruito dalla `main` su GitHub Pages.

Ad ogni commit su `main`, le pagine all'indirizzo <https://luca-heltai.github.io/nmpde/> vengono
rigenerate in modo automatico usando la *github action* qui sotto.

[![deploy-book](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml/badge.svg)](https://github.com/luca-heltai/nmpde/actions/workflows/deploy.yaml)
