import csv
import os
from datetime import datetime

from src.experiment.results import TrainingResult


class ExperimentLogger:
    """
    Unified experiment logger. Creates the output directory tree and appends
    one row per training run to a single experiment_log.csv.

    Output tree:
        {output_dir}/{experiment_name}/
            experiment_log.csv
            loss_histories/{algo}/{function}/
            predictions/{algo}/{function}/
            regions/{algo}/{function}/
    """

    _CSV_FIELDS = [
        "timestamp", "algorithm", "group", "function", "hidden_arch",
        "variation", "activation", "loss_function",
        "final_loss", "total_iters", "runtime_s",
        # algo-specific (empty when not applicable)
        "pop_size", "regions_discovered", "num_branches",
    ]

    def __init__(self, output_dir: str, experiment_name: str):
        self.root = os.path.join(output_dir, experiment_name)
        self._csv_path = os.path.join(self.root, "experiment_log.csv")
        self._csv_ready = False
        os.makedirs(self.root, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Path helpers
    # ──────────────────────────────────────────────────────────────────────────

    def arch_str(self, result: TrainingResult) -> str:
        return f"[{','.join(map(str, result.hidden_architecture))}]"

    def file_prefix(self, result: TrainingResult) -> str:
        return f"{result.variation}_{self.arch_str(result)}_{result.loss_function}"

    def plot_dir(self, folder: str, result: TrainingResult) -> str:
        return os.path.join(self.root, folder, result.algorithm, result.function)

    def get_plot_path(self, result: TrainingResult, folder: str, descriptor: str, individual: bool = False) -> str:
        """
        Returns the full save path for a plot file and ensures the parent
        directory exists.

        individual=True places the file in an _individual/ subfolder so grouped
        plots occupy the top-level directory.

        Example:
            get_plot_path(result, "regions", "regions_pre", individual=True)
            → experiments/results/exp_X/regions/bab/g_l/_individual/disc_[16,8,4]_regions_pre.png
        """
        d = self.plot_dir(folder, result)
        if individual:
            d = os.path.join(d, "_individual")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.file_prefix(result)}_{descriptor}.png")

    def get_grouped_plot_path(self, result: TrainingResult, folder: str, descriptor: str) -> str:
        """Path for a grouped plot that covers multiple variations (no variation prefix)."""
        d = self.plot_dir(folder, result)
        os.makedirs(d, exist_ok=True)
        prefix = f"combined_{self.arch_str(result)}_{result.loss_function}"
        return os.path.join(d, f"{prefix}_{descriptor}.png")

    def get_bab_iter_path(self, result: TrainingResult, iteration: int) -> str:
        d = os.path.join(self.plot_dir("predictions", result), "_individual")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{self.file_prefix(result)}_iter_{iteration:02d}.png")

    # ──────────────────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────────────────

    def log_result(self, result: TrainingResult, group_name: str = "") -> None:
        """Append one row to experiment_log.csv."""
        meta = result.metadata
        row = {
            "timestamp":          datetime.now().isoformat(),
            "algorithm":          result.algorithm,
            "group":              group_name,
            "function":           result.function,
            "hidden_arch":        self.arch_str(result),
            "variation":          result.variation,
            "activation":         "relu" if result.base_activation else "elu",
            "loss_function":      result.loss_function,
            "final_loss":         result.final_loss,
            "total_iters":        meta.get("generations") or meta.get("epochs_run") or meta.get("total_generations") or "",
            "runtime_s":          f"{result.runtime_seconds:.2f}",
            "pop_size":           meta.get("pop_size", ""),
            "regions_discovered": meta.get("regions_discovered", ""),
            "num_branches":       meta.get("num_branches", ""),
        }

        mode = "a" if self._csv_ready else "w"
        with open(self._csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
            if not self._csv_ready:
                writer.writeheader()
                self._csv_ready = True
            writer.writerow(row)
