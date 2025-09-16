# Project Overview
This repository aims at benchmarking and optimizing the DenseNet architecture. Benchmarking of the model is done using Pytorch Profiler. The key parameters monitored are RAM usage, VRAM usage for GPU, latency and throughput of the model on the device. The model is optimised to reduce the memory usage and latency of the model inference. Optimisation techniques explored on the model are quantization techniques, pruning and ONNX deployment.

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
After the run is complete, results are stored in folder ``./test_results`` in ``benchmark_results.csv``.

# Optimization Approaches
Optimization approaches explored are quantization of model, structured and unstructured pruning of model, and ONNX deployment of model.
## Quantization
Quantization techniques are techniques for performing computations and storing tensors at lower bit-widths than floating point precision. In this, model is converted from high precision (float32) numbers to low precision (int8) numbers. 
## Pruning
Pruning is the practice of removing the weights or layers in a neural network which does not have importance. This way, the size and complexity of the model can be reduced. Two kinds of pruning exists: structured pruning and unstructured pruning. Structured pruning refers to removal of entire block of parameters like filters for model size reduction. Unstructured pruning removes individual parameter across the network, based on their small magnitudes.
## ONNX
ONNX (Open Neural Network Exchange) is an open standard format for representing machine learning models. Machine learning built on frameworks like Pytorch and Tensorflow can be converted to ONNX format for faster inference.

