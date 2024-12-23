# Parallel-Programming-with-Machine-Learning
Parallel Programming with Machine Learning with OpenAcc

## How to ompile and execute my program to get the expected output on the cluster:
This project was conducted on the cluster of CUHK(SZ), on the cluster:
```bash
# On the cluster:
# Compile:
cd /path/to/project4
mkdir build && cd build
cmake ..
make
# Execute:
cd /path/to/project4
sbatch ./test.sh
```

## Brief Introduction of Codes:

**1. In `mlp_sequential.cpp` and `ops_sequential.cpp`:**

This part aims to train MNIST with MLP in a sequential way.

**2. In `mlp_openacc_kernel.cpp` and `ops_openacc_kernel.cpp`:**

This part aims to Accelerate Mlp with Kernel

**3. In `mlp_openacc_fusion.cpp` and `ops_openacc_fusion.cpp`:**

This part aims to Accelerate Mlp with Fusion
