.PHONY: train data test lint

data:
	python -m data/make_dataset_text

train:
	python -m src.rnn_pipeline.training.train

test:
	pytest tests/ -v

lint:
	black src/ tests/
	isort src/ tests/




