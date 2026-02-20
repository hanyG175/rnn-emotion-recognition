.PHONY: train data test lint

data:
	python -m src.rnn_pipeline.data.make_dataset_text

train:
	python -m src.rnn_pipeline.training.train

evaluate:
	python -m src.rnn_pipeline.evaluation.evaluate

test:
	pytest tests/ -v

lint:
	black src/ tests/
	isort src/ tests/




