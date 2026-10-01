PYTHON ?= python

.PHONY: install check notebook

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) test_environment.py

notebook:
	$(PYTHON) -m jupyter lab
