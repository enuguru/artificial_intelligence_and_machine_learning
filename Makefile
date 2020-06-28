.PHONY: clean clean-pyc update_conda

SHELL=/bin/zsh

all: update_conda

update_conda: .requirements_updated	

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean: clean-pyc
	rm -f MANIFEST
	rm -rf build dist
	
.requirements_updated: environment.yml
	conda env update -f environment.yml
	touch .requirements_updated
