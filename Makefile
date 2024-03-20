fmt:
	autoflake .
	isort .
	black .

test:
	pytest .

test-train:
	pytest tests/test_framework.py -k "test_train_framework" --capture=no
