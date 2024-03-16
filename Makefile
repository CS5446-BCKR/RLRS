fmt:
	autoflake .
	isort .
	black .

test:
	pytest .
