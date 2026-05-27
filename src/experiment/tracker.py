import csv
import json
import os
import torch
import numpy as np

class ExperimentTracker:
    """Unified logger for tracking experiment metrics, saving models and configuring output directories."""
    
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        self.experiment_name = experiment_name
        self.base_dir = base_dir
        self.output_dir = os.path.join(self.base_dir, self.experiment_name)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        self.metrics = []
        self.log_path = os.path.join(self.output_dir, "logs.csv")
        self._csv_initialized = False

    def log_epoch(self, epoch: int, loss: float, additional_metrics: dict = None):
        """Log a single training step."""
        entry = {"epoch": epoch, "loss": loss}
        if additional_metrics:
            entry.update(additional_metrics)
            
        self.metrics.append(entry)
        
    def to_csv(self):
        """Write all tracked metrics to a CSV file."""
        if not self.metrics:
            return
            
        keys = self.metrics[0].keys()
        mode = 'a' if self._csv_initialized else 'w'
        
        with open(self.log_path, mode, newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            if not self._csv_initialized:
                dict_writer.writeheader()
                self._csv_initialized = True
            dict_writer.writerows(self.metrics)
            
        self.metrics = [] # Clear memory after flush
        
    def log_final_result(self, result: dict):
        """Save final metrics/configurations as JSON."""
        path = os.path.join(self.output_dir, "final_results.json")
        with open(path, 'w') as f:
            json.dump(result, f, indent=4)
            
    def save_checkpoint(self, model: torch.nn.Module, filename: str = "best_model.pth"):
        """Save model checkpoint."""
        torch.save(model.state_dict(), os.path.join(self.output_dir, filename))
