JUPYTER_BOOK ?= $(if $(wildcard jupyterbook/bin/jupyter-book),jupyterbook/bin/jupyter-book,jupyter-book)
BUILD_DIR ?= build
CMAKE ?= cmake
CTEST ?= ctest
CMAKE_GENERATOR ?= Ninja
PYTHON ?= python3
PORT ?= 8000

.PHONY: all configure labs test assets build site serve clean latex show

all: assets site

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -G "$(CMAKE_GENERATOR)" \
		-DNMPDE_BUILD_LABS=ON -DNMPDE_BUILD_TESTS=ON

labs: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel

test: labs
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

assets: test

site:
	cd notes && BASE_URL="$${BASE_URL:-}" ../$(JUPYTER_BOOK) build --html --strict --ci --check-links

build: site

serve: site
	@echo "Serving notes/_build/html at http://127.0.0.1:$(PORT)/"
	cd notes/_build/html && $(PYTHON) -m http.server "$(PORT)" --bind 127.0.0.1

show: site
	cd notes && ../$(JUPYTER_BOOK) start

latex:
	cd notes && ../$(JUPYTER_BOOK) build --pdf

clean:
	if test -d "$(BUILD_DIR)"; then $(CMAKE) --build $(BUILD_DIR) --target clean; fi
	cd notes && ../$(JUPYTER_BOOK) clean --all --yes

lab-%: configure
	$(CMAKE) --build $(BUILD_DIR) --target lab-$* --parallel
