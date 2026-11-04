JUPYTER_BOOK ?= $(if $(wildcard jupyterbook/bin/jupyter-book),jupyterbook/bin/jupyter-book,jupyter-book)
BUILD_DIR ?= build
CMAKE ?= cmake
CTEST ?= ctest
CMAKE_GENERATOR ?= Ninja
PYTHON ?= python3
PORT ?= 8000
REGENERATE_FIGURES ?= 0
export NMPDE_REGENERATE_FIGURES = $(REGENERATE_FIGURES)

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
	$(MAKE) pacman-plots pacman-images

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

# Lecture 6: independent reproduction of the harmonic pacman experiment.
PACMAN_BUILD_DIR ?= build/pacman-release
PVPYTHON ?= pvpython
.PHONY: pacman pacman-plots pacman-images
pacman:
	$(CMAKE) -S . -B $(PACMAN_BUILD_DIR) -DCMAKE_BUILD_TYPE=Release \
		-DNMPDE_BUILD_LABS=OFF -DNMPDE_BUILD_DEALII_LABS=OFF -DNMPDE_BUILD_TESTS=ON
	$(CMAKE) --build $(PACMAN_BUILD_DIR) --target nmpde_pacman_tests --parallel 2
	$(CTEST) --test-dir $(PACMAN_BUILD_DIR) -L pacman --output-on-failure

pacman-plots:
	$(PYTHON) tests/pacman/plot_convergence.py

pacman-images:
	@if $(PYTHON) tests/figure_policy.py --needs paraview; then \
		$(PVPYTHON) --disable-registry tests/pacman/render_paraview.py; \
	else \
		echo "Both ParaView images already exist; reusing them."; \
	fi
