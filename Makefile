fmt:
	autoflake .
	isort .
	black .

test:
	pytest .

mock-train-movie:
	pytest tests/test_framework.py -k "test_train_framework" --capture=no --slow

mock-train-food:
	pytest tests/test_framework.py -k "test_ayampp_train_framework" --capture=no --slow

mlflow:
	mlflow server --host 127.0.0.1 --port 8080
