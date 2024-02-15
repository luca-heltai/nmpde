build:
	jupyter-book build notes/
publish:
	ghp-import -n -p -f notes/_build/html
show: build
	(cd notes/_build/html/ && open index.html &)
latex:
	jupyter-book build notes/ --builder pdflatex
clean:
	jupyter-book clean notes/
