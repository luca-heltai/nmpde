JUPYTER_BOOK ?= $(if $(wildcard jupyterbook/bin/jupyter-book),jupyterbook/bin/jupyter-book,jupyter-book)

.PHONY: build clean latex publish show

build:
	$(JUPYTER_BOOK) build notes/

publish:
	ghp-import -n -p -f notes/_build/html

show: build
	(open notes/_build/html/index.html &)

latex:
	$(JUPYTER_BOOK) build notes/ --builder pdflatex

clean:
	$(JUPYTER_BOOK) clean notes/
