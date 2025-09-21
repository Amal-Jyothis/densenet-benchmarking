import yaml
import sys, traceback

from steps.data_input import data_input
from steps.load_base_model import load_model
from steps.benchmarking import benchmarking
from steps.model_optimize import model_optimise
from steps.evaluation import model_evaluation
from steps.model_size_calc import model_size
from steps.saving_data import DataSaving
from steps.logger_file import Logger

logger = Logger(__name__)

def benchmark_pipeline(config_path: str, args):
    
    try:
        '''
        Reading configuration data from config.yaml file
        '''
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        '''
        Initializing Datasaving class for saving models and output
        '''
        save_data = DataSaving(
            args.output_dir, 
            config["data_storage"]["result_csv_file_name"]
        )

        '''
        Carrying out benchmarking of models for list of batch sizes and optimisation techniques
        '''
        for optimization_method in config["model"]["optimization_criteria"]:

            for batch_size in config["data"]["batch_size"]:
                if args.gpu_enabled == 'true':
                    device = 'cuda'
                else:
                    device = 'cpu'

                logger.info(f"-------Starting workflow for Optimization Method: {optimization_method} and Batch Size: {batch_size} on device: {device}-------")

                '''
                Ingesting input data. 
                Returns the dataloader
                '''
                dataloader = data_input(
                    config["data"]["name"],
                    config["data"]["root_path"], 
                    config["data"]["image_size"], 
                    batch_size
                    )

                '''
                Loading the base model for benchmarking. 
                Returns the base model.
                '''
                model = load_model(
                    config["model"]["name"]
                    )

                '''
                Applying optimisation technique on the base model. 
                Saves the model to model_output_path and returns the path on which the model is saved.
                '''
                optimised_model_path = model_optimise(
                    model, 
                    dataloader,
                    optimization_method,
                    save_data,
                    args.output_dir
                    )
                
                '''
                Calculates the size of the base model or optimised model.
                Returns the model size.
                '''
                model_size_mb = model_size(model_path=optimised_model_path)

                '''
                Carrying out benchmarking of the model
                Stores the log for tensorboard and reports from pytorch profiler to tensorboard_logs_path and profile_report_output_path. 
                Returns the benchmarking metrics for the model
                '''
                metrics = benchmarking(
                    optimised_model_path=optimised_model_path,
                    dataloader=dataloader,
                    tensorboard_logs_path=config["data_storage"]["tensorboard_logs_path"],
                    optimization_method=optimization_method,
                    device=device
                    )

                '''
                Evaluates the accuracy of the model on the input dataloader.
                Takes path to model and dataloader as input and returns the accuracy
                top_one_accuracy: top-1 accuracy
                top_five_accuracy: top-5 accuracy
                '''
                top_one_accuracy = model_evaluation(
                    optimised_model_path=optimised_model_path, 
                    dataloader=dataloader, 
                    top_N=1,
                    device=device,
                    optimization_method=optimization_method
                    )
                
                top_five_accuracy = model_evaluation(
                    optimised_model_path=optimised_model_path,
                    dataloader=dataloader,
                    top_N=5,
                    device=device,
                    optimization_method=optimization_method
                )

                '''
                Accumulates the results for storing to result_csv_file_name
                '''
                output = {
                    "model_name": config["model"]["name"],
                    "batch_size": batch_size,
                    "optimization_technique": optimization_method,
                    "device": device,
                    "model_size_mb": model_size_mb,
                    "accuracy_top_1": top_one_accuracy,
                    "accuracy_top_5": top_five_accuracy
                }

                '''
                Results stored to result_output_path
                '''                
                save_data.save_output(
                    output,
                    metrics
                )

    except Exception as e:
        _, _, tb = sys.exc_info()
        line_no = traceback.extract_tb(tb)[-1][1]
        logger.error(f'Error in pipeline: {e} at line {line_no}')

