from steps.logger_file import Logger
from torch.utils.data import DataLoader

from torchvision import transforms, datasets
import sys, traceback
import torch

logger = Logger(__name__)

def data_input(
        name: str,
        root_path: str, 
        image_size: list,
        batch_size: int) -> DataLoader:
    
    '''
    Collects data with given folder path or url path of the dataset.
    Arg: 
        data_path: path to the dataset
        image_size: size of the input images for the model
        batch_size: batch size for the input dataset
    return:
        dataloader: collected dataset

    '''

    try:
        '''
        Defining image transformation on the input data
        '''
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        '''
        Loading the dataset
        '''
        if name == "STL10":
            dataset = datasets.STL10(root_path, transform=image_transform, download=True)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(42))
        elif name == "CIFAR10":
            dataset = datasets.CIFAR10(root_path, transform=image_transform, download=True)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(42))
        else:
            raise ValueError("Dataset {} not supported.".format(name))
        return dataloader
    
    except Exception as e:
        _, _, tb = sys.exc_info()
        line_no = traceback.extract_tb(tb)[-1][1]
        logger.error(f'Error while collecting data: {e} at line {line_no}')
        raise e

