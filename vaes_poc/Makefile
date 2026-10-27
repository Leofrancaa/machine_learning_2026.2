#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = vaes_poc
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = uv run python
# Versão dos resultados (pastas models/rsna_pa_<VERSION>...) e nome da execução no MLflow
VERSION ?= v2
LABEL ?= manual

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format



## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Download RSNA from Kaggle and cache the images at 128 px (needs a Kaggle token)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m module_vae.dataset


## Full pipeline on PA exams: train -> calibrate -> evaluate -> compare
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m module_vae.main --version $(VERSION) --label $(LABEL)


## Same pipeline with AP and PA exams, to expose the AP/PA shortcut
.PHONY: train_ap_pa
train_ap_pa:
	$(PYTHON_INTERPRETER) -m module_vae.main --views AP,PA --version $(VERSION) --label $(LABEL)


## Quick check of the pipeline (2 epochs, 1 seed, no baselines), saved as version "quick"
.PHONY: quick
quick:
	$(PYTHON_INTERPRETER) -m module_vae.main --epochs 2 --n-seeds 1 --no-baselines \
		--version quick --overwrite --label quick


## Score test exams with the saved model
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m module_vae.inference


## Export the demo data (interfaces/public/data) from the trained PA and AP+PA models
.PHONY: export
export:
	$(PYTHON_INTERPRETER) -m module_vae.export


## Open the MLflow UI to compare runs (http://127.0.0.1:5001)
.PHONY: mlflow
mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001


## Latent size and beta experiments (needs `make train` first)
.PHONY: ablation
ablation:
	$(PYTHON_INTERPRETER) -m module_vae.modeling.ablation


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
