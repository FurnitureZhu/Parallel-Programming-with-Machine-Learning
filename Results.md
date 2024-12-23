# Performance Results

## Sequential (Optimized with -O2)

**Training two-layer neural network with 400 hidden units**

| Epoch | Accuracy Rate | Training Time |
|-------|---------------|---------------|
| 1     | 92.040%       | 48044 ms      |
| 2     | 93.660%       | 47971 ms      |
| 3     | 94.570%       | 47976 ms      |
| 4     | 95.380%       | 47966 ms      |
| 5     | 95.880%       | 47967 ms      |
| 6     | 96.310%       | 48000 ms      |
| 7     | 96.710%       | 47958 ms      |
| 8     | 96.840%       | 47975 ms      |
| 9     | 97.060%       | 47992 ms      |
| 10    | 97.270%       | 47988 ms      |

**Execution Time:** 524582 milliseconds


## OpenACC Kernel

**Training two-layer neural network with 400 hidden units**

| Epoch | Accuracy Rate | Training Time |
|-------|---------------|---------------|
| 1     | 92.040%       | 9849 ms       |
| 2     | 93.670%       | 8920 ms       |
| 3     | 94.580%       | 8942 ms       |
| 4     | 95.360%       | 8970 ms       |
| 5     | 95.910%       | 8944 ms       |
| 6     | 96.310%       | 9033 ms       |
| 7     | 96.710%       | 9006 ms       |
| 8     | 96.850%       | 8993 ms       |
| 9     | 97.080%       | 9043 ms       |
| 10    | 97.270%       | 9063 ms       |

**Execution Time:** 94768 milliseconds


## OpenACC Fusion

**Training two-layer neural network with 400 hidden units**

| Epoch | Accuracy Rate | Training Time |
|-------|---------------|---------------|
| 1     | 91.770%       | 7494 ms       |
| 2     | 93.380%       | 6399 ms       |
| 3     | 94.290%       | 6278 ms       |
| 4     | 95.070%       | 6309 ms       |
| 5     | 95.620%       | 6331 ms       |
| 6     | 96.030%       | 6387 ms       |
| 7     | 96.430%       | 6386 ms       |
| 8     | 96.570%       | 6414 ms       |
| 9     | 96.800%       | 6434 ms       |
| 10    | 96.990%       | 6438 ms       |

**Execution Time:** 67685 milliseconds

