PYTHON_MODULES:=$(shell python -c "import pkgutil; modules = [name if ispkg else f'{name}.py' for _, name, ispkg in pkgutil.iter_modules(['.'])]; print(' '.join(modules), end='')")
PIP_REPOSITORY=pypi

.PHONY: modules
modules:
	@echo $(PYTHON_MODULES)

.PHONY: lint
lint:
	pylint $(PYTHON_MODULES)

.PHONY: analyze
analyze:
	mypy $(PYTHON_MODULES)

.PHONY: format
format:
	isort --check $(PYTHON_MODULES) && black --check $(PYTHON_MODULES)

.PHONY: black
black:
	black $(PYTHON_MODULES)

.PHONY: isort
isort:
	isort $(PYTHON_MODULES)

.PHONY: install-dev
install-dev:
	pip install -e .[dev]

.PHONY: install
install:
	pip install .

.PHONY: test
test:
	pytest $(addprefix --cov=,$(PYTHON_MODULES))

.PHONY: build
build:
	python setup.py sdist

.PHONY: push
push: build
	twine upload dist/* -r $(PIP_REPOSITORY)