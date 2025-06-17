files=(*.py)
export OVERRIDE_UNCLEAN_REPO=1

pylint "${files[@]}"
mypy "${files[@]}"
pytest "${files[@]}"
