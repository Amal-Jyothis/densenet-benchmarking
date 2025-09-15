import torch, csv, os
from steps.logger_file import Logger

logger = Logger(__name__)

class DataSaving():
    def __init__(self, file_save_path, file_name, experiment_id = None):
        
        try:
            self.experiment_id = experiment_id
            
            self.file_save_path = file_save_path
            self.file_name = file_name
            
            if not os.path.exists(self.file_save_path):
                    os.makedirs(self.file_save_path)

            self.file_save_path_with_name = os.path.join(self.file_save_path, file_name+".csv")
                

            with open(self.file_save_path_with_name, "w") as f:
                pass
        except Exception as e:
            logger.error(f'Error while creating csv file for output: {e}')
            raise e

    def save_model(self, model, model_save_path, model_name):

        try:
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            if model_name == None:
                model_name = "BaseModel"
            
            if self.experiment_id == None:
                model_save_path_with_name = os.path.join(model_save_path, model_name+"model.pth")
                model_save_path_with_name = os.path.normpath(model_save_path_with_name)
            
            else:        
                model_save_path_with_name = os.path.join(model_save_path, model_name+self.experiment_id+"model.pth")
                model_save_path_with_name = os.path.normpath(model_save_path_with_name)

            torch.save(
                model, 
                model_save_path_with_name
            )
            return model_save_path_with_name

        except Exception as e:
            logger.error(f'Error while saving model: {e}')

    def save_output(self, output, metrics, experiment_id = None):

        try:
            output.update(metrics)
            with open(self.file_save_path_with_name, "a", newline="") as f:
                w = csv.DictWriter(f, output.keys())
                w.writeheader()
                w.writerow(output)

        except Exception as e:
            logger.error(f'Error while saving results: {e}')