# Project Overview
This repository aims at benchmarking and optimizing the DenseNet architecture. Validation set of ImageNet dataset is being used as the input for the model with 1024 samples. Benchmarking of the model is done using Pytorch Profiler. The key parameters monitored are RAM usage, VRAM usage for GPU, latency and throughput of the model on the device. The model is optimised to reduce the memory usage and latency of the model inference. Optimisation techniques explored on the model are pruning, quantization and model deployment in ONNX format.

# Setup Instructions and Usage
Requirements:
```
Docker
```
Run
```
git clone <your-repo-url>
cd <your-repo-name>
chmod +x build_and_run.sh
./build_and_run.sh --output-dir ./test_results
```
After the run is complete, results are stored in folder ``./test_results`` in ``benchmark_results.csv``. The detailed summary of resource allocation can be seen during the run at ``localhost:6006``

# Optimization Approaches
Optimization approaches explored are pruning of model, and ONNX deployment of model.
## Pruning
Pruning is the practice of removing the weights or layers in a neural network which does not have importance. This way, the size and complexity of the model can be reduced. In the case, around 1% pruning of convolution layers has been done. Further pruning can lead to drastic drop in accuracy. Pruning has to be carried out carefully.
## ONNX
ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models. Machine learning built on frameworks like Pytorch and Tensorflow can be converted to ONNX format for faster inference.

# Results
The metrics reported for each methods are: RAM usage, VRAM usage for GPU, latency, throughput of the model, top-1 accuracy and top-5 accuracy for batch sizes of [1, 4, 8, 16, 32]. The results for an experiment run on NVIDIA RTX A5000 GPU is shown in the table.

|model_name |batch_size|optimization_technique|device|model_size_mb|accuracy_top_1|accuracy_top_5|ram_usage_mb|vram_usage_mb|latency_ms_per_batch|throughput_samples_per_sec|
|-----------|----------|----------------------|------|-------------|--------------|--------------|------------|-------------|--------------------|--------------------------|
|Densenet121|1         |None                  |cuda  |31.1         |88.4          |97.1          |4.6         |241.8        |94.5                |10.6                      |
|Densenet121|4         |None                  |cuda  |31.1         |88.4          |97.1          |18.4        |972.7        |188.3               |21.2                      |
|Densenet121|8         |None                  |cuda  |31.1         |88.4          |97.1          |36.8        |1945.1       |195.8               |40.9                      |
|Densenet121|16        |None                  |cuda  |31.1         |88.4          |97.1          |73.5        |3871.3       |153.1               |104.5                     |
|Densenet121|32        |None                  |cuda  |31.1         |88.4          |97.1          |147         |7720         |325.4               |98.3                      |
|Densenet121|1         |Pruning               |cuda  |31           |65.5          |87.6          |4.6         |241.8        |42.4                |23.6                      |
|Densenet121|4         |Pruning               |cuda  |31           |65.5          |87.6          |18.4        |948.3        |225.3               |17.8                      |
|Densenet121|8         |Pruning               |cuda  |31           |65.5          |87.6          |36.8        |1901.1       |165.9               |48.2                      |
|Densenet121|16        |Pruning               |cuda  |31           |65.5          |87.6          |73.5        |3774.3       |225.6               |70.9                      |
|Densenet121|32        |Pruning               |cuda  |31           |65.5          |87.6          |147         |7525.9       |339.8               |94.2                      |
|Densenet121|1         |ONNX                  |cuda  |1.1          |88.4          |97.1          |4.6         |0            |39.6                |25.2                      |
|Densenet121|4         |ONNX                  |cuda  |1.1          |88.4          |97.1          |18.4        |0            |90.2                |44.4                      |
|Densenet121|8         |ONNX                  |cuda  |1.1          |88.4          |97.1          |36.9        |0            |104.9               |76.3                      |
|Densenet121|16        |ONNX                  |cuda  |1.1          |88.4          |97.1          |73.7        |0            |31.9                |500.9                     |
|Densenet121|32        |ONNX                  |cuda  |1.1          |88.4          |97.1          |147.5       |0            |53.5                |597.8                     |

The performance improvement on pruning methods are not effective as much in terms of accuracy of inference and throughput while ONNX deployment of the model. Peak VRAM usage increases with batch size, while reducing the batch size increases the overall inference time. The batch size can be decided based on the capability of the available GPU.

# Limitations and Improvement
- Quantization of the model.
- Logging of VRAM usage during ONNX deployment
- Multiple runs need to be carried out to get the mean and deviation of inference and throughput.
- Serverless deployment of the inference.