import json
import csv
import threading
from pathlib import Path
from datetime import datetime
from .config import LOGS_DIR

class ExperimentTracker:
    def __init__(self, experiment_name=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{experiment_name}_{timestamp}" if experiment_name else timestamp
        self.run_dir = LOGS_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_path = self.run_dir / "experiment_config.json"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        
        print(f"Explore experiment logs at: {self.run_dir}")

    def log_config(self, config_data):
        """
        Saves configuration dictionary to JSON.
        config_data: dict containing hyperparameters, model info, etc.
        """
        def _write_json():
            # Convert non-serializable objects (like paths) to strings
            serializable_config = self._make_serializable(config_data)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_config, f, indent=4)
        
        # Parallelize I/O
        threading.Thread(target=_write_json).start()

    def log_metric(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        """
        Logs epoch metrics to CSV.
        """
        def _write_csv():
            file_exists = self.metrics_path.exists()
            with open(self.metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
                writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])

        # Parallelize I/O
        threading.Thread(target=_write_csv).start()

    def _make_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, '__class__') and hasattr(obj, '__dict__'):
             # Try to capture object representation if it's a class instance like Optimizer
             return str(obj)
        return obj

    def get_run_dir(self):
        return self.run_dir
