.PHONY: train test lint
   
   train:
       python -m src.rnn_pipeline.training.train
   
   test:
       pytest tests/ -v
   
   lint:
       black src/ tests/
       isort src/ tests/