.PHONY: test test_coverage
all: test pre-commit clean build publish_test publish_prod

test:
	python -m unittest -v
test_coverage:
	coverage run -m unittest && coverage report -m
pre-commit:
	pre-commit run --all-files
clean:
	rm -r build dist metaflow_folder/*.egg-info
install:
	python -m pip install --editable "./[dev]"
build:
	python -m build --sdist --wheel ./
publish_test:
	twine upload dist/* -r testpypi --username=__token__ --password=${TEST_PYPI_TOKEN} --verbose
publish_prod:
	twine upload dist/* -r pypi --username=__token__ --password=${TEST_PROD_TOKEN} --verbose
