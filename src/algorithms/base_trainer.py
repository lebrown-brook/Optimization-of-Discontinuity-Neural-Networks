from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """Abstract base class for all training algorithms."""
    
    def __init__(self, model, x_train, y_train, loss_name, **kwargs):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.loss_name = loss_name
        self.loss_history = []
        
    @abstractmethod
    def train(self):
        """Execute the training process."""
        pass
    
    def get_loss_history(self):
        """Returns the logged loss history over epochs/generations."""
        return self.loss_history
    
    def get_best_model(self):
        """Returns the best model found during training."""
        return self.model
