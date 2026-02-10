# MLflow/W&B integration (NEW)

import mlflow
from pathlib import Path
 
class ExperimentTracker:
    def __init__(self, experiment_name: str,
                 tracking_uri: str = "mlruns"):
        # "mlruns" → local folder. Use "http://server:5000" in prod
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
 
    def start_run(self, run_name=None):
        mlflow.start_run(run_name=run_name)
 
    def log_params(self, params: dict):
        # MLflow needs flat dict — nested config must be flattened first
        mlflow.log_params(_flatten(params))
 
    def log_metrics(self, metrics: dict, step: int):
        # step = epoch index — provides the time axis in the MLflow UI
        mlflow.log_metrics(metrics, step=step)
 
    def log_artifact(self, path: str):
        if Path(path).exists(): mlflow.log_artifact(path)
 
    def end_run(self):
        mlflow.end_run()
 
# Internal: {"training": {"lr": 0.001}} → {"training.lr": 0.001}
def _flatten(d, parent="", sep=".") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        items.update(_flatten(v, key) if isinstance(v, dict) else {key: v})
    return items
