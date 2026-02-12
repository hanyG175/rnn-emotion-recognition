.PHONY: train test lint

   data:
        python src.rnn_pipeline.data.make_dataset.py

   train:
        python -m src.rnn_pipeline.training.train
   
   test:
        pytest tests/ -v
   
   lint:
        black src/ tests/
        isort src/ tests/




