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
versionate: sono ricreate automaticamente sia localmente sia in CI.

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
