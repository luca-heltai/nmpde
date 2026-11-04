# Metodi Numerici per Equazioni alle Derivate Parziali

[Libro del corso](https://luca-heltai.github.io/nmpde/)

Il libro web è scritto in MyST Markdown e viene costruito con Jupyter Book 2.
Il workflow GitHub Actions compila e testa i laboratori nell’immagine
`dealii/dealii:v9.7.1-noble`, genera le figure del corso con la testsuite GTest,
costruisce il libro e lo pubblica su GitHub Pages.

## Libro web

Sul computer del docente l’ambiente Python usa `~/anaconda3/bin/python3`:

```bash
source ./start.sh
make site
```

Il comando `make site` costruisce il sito HTML in `notes/_build/html/`.
Per la build completa, compresi compilazione dei laboratori, test e figure:

```bash
make all
```

Le figure generate dai test sono in `notes/assets/generated/` e non vengono
versionate. I generatori mantengono gli output esistenti e producono soltanto
quelli mancanti, sia localmente sia in CI.

## Laboratori deal.II

È disponibile un devcontainer basato sull’immagine ufficiale:

```text
dealii/dealii:v9.7.1-noble
```

Dopo l’apertura del repository nel container:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

I programmi sono costruiti da un unico `CMakeLists.txt` alla radice. Per
eseguire un laboratorio, ad esempio:

```bash
./build/bin/lab-01
```

La testsuite si trova in `tests/`; oltre a verificare la presenza dei materiali
dei laboratori, genera le figure SVG usate nelle lecture e nei laboratori.

## Struttura

- `notes/`: libro web e contenuti pubblicati;
- `labs/`: README e sorgenti C++ dei laboratori;
- `tests/`: testsuite GTest e generatori di figure;
- `.devcontainer/`: ambiente riproducibile deal.II;
- `.github/workflows/deploy.yaml`: CI, build e deploy GitHub Pages.

Le lezioni e i laboratori futuri restano fuori dalla TOC finché non vengono
rilasciati durante il corso.

## Esperimento pacman della lezione 6

Il codice autonomo in `tests/pacman/` riproduce gli esperimenti di convergenza
con cinque aperture, genera CSV e file VTU/PVD, e verifica errori, residuo e
quadratura. Le istruzioni complete sono in [tests/pacman/README.md](tests/pacman/README.md).
Dalla radice del repository, nell'ambiente deal.II:

```bash
make pacman
make pacman-plots PYTHON=python3
make pacman-images PVPYTHON=/path/to/pvpython
```

Il secondo comando richiede NumPy e Matplotlib (`requirements.txt`); il terzo
usa ParaView quando mancano le immagini. `make all` verifica la disponibilità
di tutti i dati, grafici e PNG e genera quelli mancanti. I PNG ParaView sono snapshot versionabili,
rigenerabili con l'ultimo comando. La lezione 6 è inclusa nella TOC pubblica.

## Rigenerazione esplicita delle figure

Inserire **`[regenerate figures]`** nel messaggio di commit oppure nel titolo o
nel corpo della PR per forzare tutti i generatori in CI, inclusi esperimenti
numerici, grafici, tabelle e PNG ParaView. Sono controllati i messaggi dei commit
del push e, per una PR, anche quelli dei commit fra base e head; il confronto
non distingue maiuscole e minuscole. La modifica di titolo/corpo della PR
riattiva il workflow. Il push del commit di merge controlla anche il titolo e
il corpo della PR associata, così una keyword nel solo corpo della PR vale
anche per la build di pubblicazione su `main`.

Senza keyword, la CI recupera gli output dalla cache e genera solo i file
mancanti o vuoti. Una cache assente o scaduta richiede di ricreare gli output
non presenti nel checkout. Gli output completi del pacman comportano test
GTest esplicitamente segnati come saltati; non attestano un nuovo calcolo.
Dopo modifiche ai generatori, alle mesh o ai parametri usare la keyword per
richiedere una nuova validazione numerica.

Per forzare localmente l'intera pipeline:

```bash
make all REGENERATE_FIGURES=1 PYTHON=python3 PVPYTHON=/path/to/pvpython
```

Per il solo esperimento numerico: `make pacman REGENERATE_FIGURES=1`.
Gli script Python accettano `--force`; i test C++ leggono
`NMPDE_REGENERATE_FIGURES=1`. In assenza di questa richiesta, gli output già
presenti conservano contenuto e data di modifica.
