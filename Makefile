.PHONY: servedocs docs help
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

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

format: ## style code using ruff format
	uv run ruff format src/rbrmoored/

format-check: ## check code style using ruff format --diff
	uv run ruff format --diff src/rbrmoored/

check: ## run ruff code checking
	uv run ruff check src/rbrmoored/

test: ## run tests quickly with the default Python
	uv run pytest

docs: ## generate documentation using pdoc
	rm -rf docs
	PDOC_ALLOW_EXEC=1 pdoc -d numpy -o docs -t .pdoc-theme-gv --math ./rbrmoored
	$(BROWSER) docs/index.html

servedocs: ## compile the docs & watch for changes
	PDOC_ALLOW_EXEC=1 pdoc -d numpy -t .pdoc-theme-gv --math rbrmoored
	# $(BROWSER) http://localhost:8080
