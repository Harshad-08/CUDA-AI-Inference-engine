# CUDA AI Inference Engine

A GPU-accelerated AI inference engine built using **C++ and CUDA**.  
This project demonstrates how neural network operations such as **matrix multiplication and activation functions** can be accelerated using parallel GPU computing.

The system benchmarks **CPU vs GPU execution** and shows significant performance improvements through CUDA kernel parallelization.

---

## Features

- GPU-accelerated matrix multiplication using CUDA kernels
- ReLU activation function implemented on GPU
- Simple neural network inference pipeline
- CPU vs GPU performance benchmarking
- Demonstrates GPU parallel programming and CUDA memory management

---

## Project Structure

```
cuda-ai-inference-engine
│
├── src
│   ├── matrix_mul.cu
│   ├── activation.cu
│   ├── gpu_kernels.cu
│   └── main.cpp
│
├── include
│   ├── matrix.h
│   └── activation.h
│
├── benchmarks
│   └── results.md
│
├── colab
│   └── inference_demo.ipynb
│
├── CMakeLists.txt
└── README.md
```

---

## Requirements

- CUDA Toolkit
- NVIDIA GPU
- C++17
- CMake

Since CUDA requires NVIDIA GPUs, this project can be executed using **Google Colab GPU runtime** if local hardware does not support CUDA.

---

## Build Instructions

Clone the repository:

```bash
git clone https://github.com/your-username/cuda-ai-inference-engine.git
cd cuda-ai-inference-engine
```

Build the project:

```bash
mkdir build
cd build
cmake ..
make
```

Run the program:

```bash
./inference_engine
```

---

## Run on Google Colab

CUDA execution requires an NVIDIA GPU.  
You can run this project using Google Colab.

Steps:

1. Open `colab/inference_demo.ipynb`
2. Enable GPU runtime  
3. Run the notebook cells

This will compile and execute the CUDA kernels on a GPU.

---

## Example Inference Pipeline

The project simulates a simple neural network inference pipeline:

```
Input
   ↓
Matrix Multiplication (CUDA)
   ↓
ReLU Activation (CUDA)
   ↓
Output
```

---

## Performance Benchmark

Matrix Size: **512 × 512**

| Implementation | Execution Time |
|----------------|---------------|
| CPU | 0.641 sec |
| GPU (CUDA) | 0.021 sec |

GPU acceleration achieved **~30× speedup** compared to CPU execution.

---

## Learning Outcomes

This project demonstrates:

- CUDA kernel development
- GPU memory management
- Parallel matrix computations
- Neural network inference operations
- CPU vs GPU performance benchmarking
- GPU acceleration techniques

---

## Author

**Harshad Magdum**  
Computer Science Student

---

## Future Improvements

- Shared memory optimized matrix multiplication
- Support for larger neural network layers
- Additional activation functions
- Tensor core optimization
- Integration with deep learning models
