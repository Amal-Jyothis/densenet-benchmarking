from steps.logger_file import Logger
from steps.saving_data import DataSaving

import torch
import warnings
from typing import Union
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

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

    if optimization_method == None:
        model_path = save_data.save_model(
            model,
            model_save_path,
            optimization_method
        )
        return model_path
    
    if optimization_method == 'UnstructuredPruning':
        logger.info(f"Carrying out unstructured pruning of the model.")
        '''
        Carrying out unstructured pruning with removal of 30% of insignificant parameters.
        Weights lying in bottom 30% on the basis of L1 norm is removed
        '''
        prune.l1_unstructured(model.features.conv0, name='weight', amount=0.3)
        prune.remove(model.features.conv0, 'weight')
        model_path = save_data.save_model(
            model,
            model_save_path,
            optimization_method
        )
        return model_path
    
    if optimization_method == 'StructuredPruning':
        logger.info(f"Carrying out structured pruning of the model.")
        '''
        Carrying out structured pruning with removal of 30% of insignificant parameters.
        Filters lying in bottom 30% on the basis of L2 norm is removed
        '''
        prune.ln_structured(model.features.conv0, name='weight', amount=0.3, n=2, dim=0) 
        prune.remove(model.features.conv0, 'weight')
        model_path = save_data.save_model(
            model,
            model_save_path,
            optimization_method
        )
        return model_path

    if optimization_method == 'Quantization':
        logger.info("Carrying out quantization of the model.")
        model.eval()
        example_images, example_labels = next(iter(dataloader))
        example_inputs = (example_images,)

        '''
        Carrying out quantization of the model. Converted from float to int8 by calibrating with sample input data to minimise inaccuracies.
        '''
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            exported = torch.export.export(model, example_inputs).module()
            quantizer = X86InductorQuantizer()
            prepared = prepare_pt2e(exported, quantizer)
            def calibrate(model, data_loader):
                torch.ao.quantization.move_exported_model_to_eval(model)
                with torch.no_grad():
                    for i, (image, _) in enumerate(data_loader):
                        if i >= 10:
                            break
                        model(image)
            calibrate(prepared, dataloader)
            quant_model = convert_pt2e(prepared)
            
        model_path = save_data.save_model(
            quant_model,
            model_save_path,
            optimization_method
        )
        return model_path
    
    if optimization_method == 'ONNX':
        logger.info("Generating ONNX model.")
        example_images, example_labels = next(iter(dataloader))
        example_inputs = (example_images,)
        model_path = model_save_path + "/model" + optimization_method + ".onnx"
        torch.onnx.export(model, example_inputs, model_path, dynamo=True)
        return model_path
    else:
        raise ValueError("Optimization method {} not supported.".format(optimization_method))
    