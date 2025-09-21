from steps.logger_file import Logger

import torch, onnxruntime, numpy, torchvision
import sys, traceback
from torch.utils.data import DataLoader

logger = Logger(__name__)

def model_evaluation(
        optimised_model_path: str, 
        dataloader: DataLoader, 
        top_N: int, 
        device: str, 
        optimization_method: str) -> float:
    '''
    Carry out the evaluation of the model.
    arg:
        model_path: folder path to the model
        dataloader: input data
        top_N: defines the evaluation criteri. eg. for top-1 accuracy, top_N=1. for top-5 accuracy, top_N=5.
        device: device on which computation happens. 'cuda' or 'cpu'
        optimization_method: optimization technique applied on the model
    return:
        accuracy of the model
    '''
    try:
        correctly_pred_case = 0
        total_cases = 0

        for i, (images, labels) in enumerate(dataloader):
            if optimization_method == "ONNX":
                if device == 'cuda':
                    ort_session = onnxruntime.InferenceSession(optimised_model_path, providers=["CPUExecutionProvider"])
                else:
                    ort_session = onnxruntime.InferenceSession(optimised_model_path, providers=["CUDAExecutionProvider"])
                output = torch.tensor(numpy.array(ort_session.run(None, {"x": images.numpy()})))
                output = torch.squeeze(output, 0).to(torch.device(device))

            else:
                model = torch.load(optimised_model_path, weights_only=False).to(torch.device(device))
                model.eval()
                output = model(images.to(torch.device(device)))
            
            pred_label_sorted = torch.sort(output, dim=-1, descending=True).indices[:, :top_N]
            matches = torch.unsqueeze(labels.to(torch.device(device)), 1) == pred_label_sorted
            correctly_pred_case += torch.sum(matches).item()
            total_cases += len(labels)
            
        accuracy = (correctly_pred_case/total_cases)*100
        logger.info(f"Accuracy of the model is {accuracy}")
        return accuracy

    except Exception as e:
        _, _, tb = sys.exc_info()
        line_no = traceback.extract_tb(tb)[-1][1]
        logger.error(f'Error while evaluating model: {e} at line {line_no}')