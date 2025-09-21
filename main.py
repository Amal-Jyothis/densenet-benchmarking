from steps.logger_file import Logger

import argparse
from fastapi import FastAPI
import uvicorn

from pipeline.benchmark_pipeline import benchmark_pipeline

logger = Logger(__name__)

if __name__ == "__main__":
    config_path = "config.yaml"
    logger.info(f"Starting....")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--gpu-enabled", type=str, default="false")
    args = parser.parse_args()

    '''
    Executing benchmark pipeline
    '''
    benchmark_pipeline(config_path, args)
