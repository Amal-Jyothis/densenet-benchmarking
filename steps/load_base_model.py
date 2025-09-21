import torchvision
import torch, torchvision
from typing import Tuple
from torch.utils.data import DataLoader

def load_model(
    model_name: str
    ) -> torch.nn.Module:
    '''
    Loads the base model for benchmarking.
    arg:
        model_name: Name of the base model
    return:
        model: Pytorch base model
    '''
    model = None
    if model_name == 'Densenet121':
        model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
        model.eval()
        return model
    else:
        raise ValueError("Model {} not supported.".format(model_name))
    