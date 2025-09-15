import logging
import sys

class Logger:
    def __init__(self, name: str, log_file: str = None, level=logging.INFO, experiment_id: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        self.experiment_id = experiment_id

    def info(self, message: str):
        if self.experiment_id:
            self.logger.info(f"[exp_id={self.experiment_id}] {message}")
        else:
            self.logger.info(message)

    def error(self, message: str):
        if self.experiment_id:
            self.logger.error(f"[exp_id={self.experiment_id}] {message}")
        else:
            self.logger.error(message)

    def debug(self, message: str):
        if self.experiment_id:
            self.logger.debug(f"[exp_id={self.experiment_id}] {message}")
        else:
            self.logger.debug(message)