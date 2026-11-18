.PHONY: all prepare validate train test

all: prepare validate train test

prepare:
	python src/prepare.py

validate:
	python src/validate_data.py

train:
	python src/train.py

test:
	python -m pytest tests/ -v
