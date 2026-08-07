.PHONY: help format format-check check test docs docs-build servedocs \
	dev-pyrsktools undev-pyrsktools update-pyrsktools
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := uv run python -c "$$BROWSER_PYSCRIPT"

help:
	@uv run python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

format: ## style code using ruff format
	uv run ruff format src/ tests/

format-check: ## check code style using ruff format --diff
	uv run ruff format --diff src/ tests/

check: ## run ruff code checking
	uv run ruff check src/ tests/

test: ## run tests quickly with the default Python
	uv run pytest

dev-pyrsktools: ## link ../pyrsktools into the venv as an editable install
	uv sync
	uv pip install -e ../pyrsktools --no-deps
	@echo
	@echo "pyrsktools is now editable from ../pyrsktools."
	@echo "Use 'uv run --no-sync ...' (or export UV_NO_SYNC=1) while working on it;"
	@echo "a plain 'uv run' or 'uv sync' restores the pinned git version."

undev-pyrsktools: ## restore pyrsktools from the pinned git source
	uv sync --reinstall-package pyrsktools

update-pyrsktools: ## pull the latest commit from the pyrsktools git branch
	uv lock --upgrade-package pyrsktools
	uv sync

docs-build: ## generate documentation using pdoc (no browser)
	rm -rf docs
	PDOC_ALLOW_EXEC=1 uv run pdoc -d numpy -o docs -t .pdoc-theme-gv --math rbrmoored

docs: docs-build ## generate documentation and open it in the browser
	$(BROWSER) docs/index.html

servedocs: ## serve the docs & watch for changes
	PDOC_ALLOW_EXEC=1 uv run pdoc -d numpy -t .pdoc-theme-gv --math rbrmoored
