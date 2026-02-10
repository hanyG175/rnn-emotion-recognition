import csv
from pathlib import Path
 
class MetricsLogger:
    def __init__(self, filepath: str = "artifacts/metrics/metrics.csv"):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" prevents blank rows on Windows
        self._file = open(path, "w", newline="")
        self._writer = None   # created lazily on first log() call
 
    def log(self, metrics: dict) -> None:
        if self._writer is None:
            # Write CSV header from first dict keys
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(metrics.keys()))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        # Flush after each row — file stays readable even if process crashes
        self._file.flush()
 
    def close(self) -> None:
        self._file.close()
