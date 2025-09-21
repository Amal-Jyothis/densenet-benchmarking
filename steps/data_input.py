from steps.logger_file import Logger
from torch.utils.data import DataLoader, Subset

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
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])

        '''
        Loading the dataset
        '''
        if name == "STL10":
            dataset = datasets.STL10(root_path, transform=image_transform, download=True)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, generator=torch.Generator().manual_seed(42))
            logger.info('Data ingestion completed')
            return dataloader
        if name == "CIFAR10":
            dataset = datasets.CIFAR10(root_path, transform=image_transform, download=True)
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(42))
            logger.info('Data ingestion completed')
            return dataloader
        if name == "ImageNet":
            dataset = datasets.ImageNet(f"{root_path}", transform=image_transform, split="val")
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, generator=torch.Generator().manual_seed(42))
            logger.info('Data ingestion completed')
            return dataloader
        else:
            raise ValueError("Dataset {} not supported.".format(name))
        
    
    except Exception as e:
        _, _, tb = sys.exc_info()
        line_no = traceback.extract_tb(tb)[-1][1]
        logger.error(f'Error while collecting data: {e} at line {line_no}')
        raise e

