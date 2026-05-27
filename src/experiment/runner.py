from abc import ABC, abstractmethod
from src.experiment.tracker import ExperimentTracker

class ExperimentRunner(ABC):
    """Base class for running experiment workflows across algorithms."""
    
    def __init__(self, config_settings, tracker: ExperimentTracker):
        self.config = config_settings
        self.tracker = tracker
        
    @abstractmethod
    def prepare_data(self):
        """Prepare datasets and loaders."""
        pass
        
    @abstractmethod
    def build_model(self):
        """Build the neural network architecture."""
        pass
        
    @abstractmethod
    def start(self):
        """Execute the full experiment workflow."""
        pass
