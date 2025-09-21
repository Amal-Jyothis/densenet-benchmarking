from steps.logger_file import Logger

from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader
import torch, onnxruntime
import sys, traceback
import numpy, torchvision

logger = Logger(__name__)

def benchmarking(
        optimised_model_path: str,
        dataloader: DataLoader, 
        tensorboard_logs_path: str, 
        optimization_method: str, 
        device: str,
        wait: int = 1,
        warmup: int = 1, 
        active: int = 4, 
        repeat: int = 4,
        number_of_steps: int = 5):
    '''
    Carries out the benchmarking of the model
    arg:
        model_path: Path to the saved model
        dataloader: Input data
        tensorboard_logs_path: path to saving tensorboard logs, 
        optimization_method: optimization method
    '''

    try:
        logger.info(f"Starting benchmarking of the model")
        if torch.cuda.is_available():
            logger.info("CUDA is available")
        activities = [ProfilerActivity.CPU]
        if device == 'cuda':
            activities += [ProfilerActivity.CUDA] 

        profiler = profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(tensorboard_logs_path),
            record_shapes=True,
            with_stack=True, 
            profile_memory=True
        )

        if optimization_method == "ONNX":
            batch_count = 0

            with profiler as prof:
                with record_function("model_inference"):
                    if device == 'cuda':
                        ort_session = onnxruntime.InferenceSession(optimised_model_path, providers=["CUDAExecutionProvider"])
                    elif device == 'cpu':
                        ort_session = onnxruntime.InferenceSession(optimised_model_path, providers=["CPUExecutionProvider"])
                    else:
                        raise ValueError("Device {} not supported.".format(device))
                    
                    for i, (images, _) in enumerate(dataloader):
                        output = output = torch.tensor(numpy.array(ort_session.run(None, {"x": images.numpy()})))
                        output = torch.squeeze(output, 0)
                        prof.step()

                        if i >= number_of_steps:
                            break
                        batch_count += 1

        else:
            batch_count = 0

            with profiler as prof:
                with record_function("model_inference"):
                    model = torch.load(optimised_model_path, weights_only=False).to(torch.device(device))
                    model.eval()
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    
                    for i, (images, _) in enumerate(dataloader):
                        output = model(images.to(torch.device(device)))
                        prof.step()

                        if i >= number_of_steps:
                            break
                        batch_count += 1

        if device == 'cuda':
            peak_vram = max(evt.device_memory_usage for evt in prof.key_averages()) / 1024**2
        else:
            peak_vram = None
        peak_cpu_mem = max(evt.cpu_memory_usage for evt in prof.key_averages()) / 1024**2

        if device == 'cuda':
            total_cuda_time_ms = sum(evt.device_time_total for evt in prof.key_averages()) / 1000.0
        else:
            total_cuda_time_ms = None
        total_cpu_time_ms = sum(evt.cpu_time_total for evt in prof.key_averages()) / 1000.0 

        if device == 'cuda':
            avg_latency = (total_cuda_time_ms/batch_count)
            avg_throughput = (dataloader.batch_size/avg_latency) * 1000

        elif device == 'cpu':
            avg_latency = (total_cpu_time_ms/batch_count)
            avg_throughput = (dataloader.batch_size/avg_latency) * 1000

        else:
            raise ValueError("Device {} not supported.".format(device))
        
        sort_by_keyword = device + "_time_total"

        metrics = {
            "ram_usage_mb": peak_cpu_mem,
            "vram_usage_mb": peak_vram,
            "latency_ms": avg_latency,
            "throughput_samples_sec": avg_throughput,
        }
        
        # logger.info(f"\n{prof.key_averages()}")

        return metrics
        
    except Exception as e:
        _, _, tb = sys.exc_info()
        line_no = traceback.extract_tb(tb)[-1][1]
        logger.error(f'Error while benchmarking: {e} at line {line_no}')
        