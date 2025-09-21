from steps.logger_file import Logger
from steps.saving_data import DataSaving

import torch
import warnings, os, sys, traceback
from typing import Union
from torch.utils.data import DataLoader
import torch_pruning as tp

logger = Logger(__name__)

def model_optimise(
    model: torch.nn.Module,
    dataloader: DataLoader, 
    optimization_method: str,
    save_data: DataSaving,
    model_save_path: str
    ) -> str:

    '''
    Applies optimization techniques to the base model and saves the optimised model
    arg:
        model: base model
        dataloader: the input data used for calibration while optimising the model
        optimization_method: optimization method to be applied to base model
        save_data: DataSaving class to save the model
        model_save_path: folder path to which the model is saved
    
    saves the model to the folder path
    return:
        model_path: path of the saved model
    '''

    try:
        model_save_path = os.path.join(model_save_path, "models")

        if optimization_method == None:
            model_path = save_data.save_model(
                model,
                model_save_path,
                optimization_method
            )
            return model_path
        
        if optimization_method == 'Pruning':
            logger.info(f"Carrying out pruning of the model.")
            '''
            Carrying out unstructured pruning with removal of insignificant parameters.
            '''
            example_inputs = torch.randn(1, 3, 224, 224)

            DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
            group = DG.get_pruning_group(model.features.conv0, tp.prune_conv_out_channels, idxs=range(0, model.features.conv0.out_channels, 6))

            if DG.check_pruning_group(group):
                group.prune()

            model.zero_grad()
            model_path = save_data.save_model(
                model,
                model_save_path,
                optimization_method
            )
            return model_path
        
        elif optimization_method == 'ONNX':
            logger.info("Generating ONNX model.")
            example_images, example_labels = next(iter(dataloader))
            example_inputs = (example_images,)
            model_path = model_save_path + "/model" + optimization_method + ".onnx"
            torch.onnx.export(model, example_inputs, model_path, dynamo=True)
            return model_path
        else:
            raise ValueError("Optimization method {} not supported.".format(optimization_method))
    
    except Exception as e:
        _, _, tb = sys.exc_info()
        line_no = traceback.extract_tb(tb)[-1][1]
        logger.error(f'Error while model optimization: {e} at line {line_no}')