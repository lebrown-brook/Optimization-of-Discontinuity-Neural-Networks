from src.models.dnn import DNN
from src.models.gadnn import GADNN
from src.models.sigmoidnn import sigmoidNN

MODEL_REGISTRY = {
    'dnn': DNN,
    'gadnn': GADNN,
    'sigmoidnn': sigmoidNN,
}

def get_model(name: str, **kwargs):
    """Retrieve and instantiate a model by its registry name."""
    if name.lower() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name.lower()](**kwargs)
