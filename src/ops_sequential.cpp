#include "ops.hpp"
#include <cstring>

const float epsilon = 1e-20;

void gemm(const float* A, const float* B, float* Out, size_t batch, size_t mn, size_t k)
{
    for (size_t i = 0; i < batch; ++i)
    {
        for (size_t j = 0; j < k; ++j)
        {
            float sum = 0.0f;
            for (size_t m = 0; m < mn; ++m)
            {
                sum += A[i * mn + m] * B[m * k + j];
            }
            Out[i * k + j] = sum;
        }
    }
}

void add_bias(float* A, float* B, const float* bias, size_t batch, size_t out_dim)
{
    for (size_t i = 0; i < batch; ++i)
    {
        for (size_t j = 0; j < out_dim; ++j)
        {
            B[i * out_dim + j] = A[i * out_dim + j] + bias[j];
        }
    }
}

void Relu(float* A, float* B, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        B[i] = (A[i] > 0.0f) ? A[i] : 0.0f;
    }
}

void Softmax(float* A, float* B, size_t batch, size_t out_dim)
{
    for (size_t i = 0; i < batch; ++i)
    {
        // Find the maximum value in the current batch to stabilize numerical calculations
        float max_val = A[i * out_dim];
        for (size_t j = 1; j < out_dim; ++j)
        {
            if (A[i * out_dim + j] > max_val)
            {
                max_val = A[i * out_dim + j];
            }
        }

        // Calculate the sum of exponentials
        float sum = 0.0f;
        for (size_t j = 0; j < out_dim; ++j)
        {
            B[i * out_dim + j] = std::exp(A[i * out_dim + j] - max_val);
            sum += B[i * out_dim + j];
        }

        // Normalize
        for (size_t j = 0; j < out_dim; ++j)
        {
            B[i * out_dim + j] /= sum + epsilon; // Add epsilon to prevent division by zero
        }
    }
}

void vector_to_one_hot_matrix(const unsigned char* A, float* B, size_t batch, size_t out_dim)
{
    std::memset(B, 0, sizeof(float) * batch * out_dim);

    for (size_t i = 0; i < batch; ++i)
    {
        unsigned char label = A[i];
        if (label < out_dim)
        {
            B[i * out_dim + label] = 1.0f;
        }
    }
}

void cross_entropy_loss(const float* A, const float* B, float* Loss, size_t batch, size_t out_dim)
{
    for (size_t i = 0; i < batch; ++i)
    {
        float loss = 0.0f;
        for (size_t j = 0; j < out_dim; ++j)
        {
            // Use log(A + epsilon) to prevent log(0)
            loss -= B[i * out_dim + j] * std::log(A[i * out_dim + j] + epsilon);
        }
        Loss[i] = loss;
    }
}

void cross_entropy_loss_grad(const float* A, const float* B, float* Grad, size_t batch, size_t out_dim)
{
    for (size_t i = 0; i < batch; ++i)
    {
        for (size_t j = 0; j < out_dim; ++j)
        {
            Grad[i * out_dim + j] = (A[i * out_dim + j] - B[i * out_dim + j]) / batch;
        }
    }
}

void update_bias(float* Bias, const float* Output_Loss, size_t batch, float lr, size_t out_dim)
{
    for (size_t j = 0; j < out_dim; ++j)
    {
        float grad = 0.0f;
        for (size_t i = 0; i < batch; ++i)
        {
            grad += Output_Loss[i * out_dim + j];
        }
        Bias[j] -= lr * grad;
    }
}

void input_grad(const float* Weight, const float* Output_Loss, float* Input, float* Grad, size_t batch, size_t in_dim, size_t out_dim)
{
    for (size_t i = 0; i < batch; ++i)
    {
        for (size_t j = 0; j < in_dim; ++j)
        {
            float grad_val = 0.0f;
            for (size_t k = 0; k < out_dim; ++k)
            {
                grad_val += Output_Loss[i * out_dim + k] * Weight[j * out_dim + k];
            }
            Grad[i * in_dim + j] = grad_val;
        }
    }
}

void update_weight(float* Weight, const float* Output_Loss, const float* Input, size_t batch, float lr, size_t in_dim, size_t out_dim)
{
    for (size_t j = 0; j < in_dim; ++j)
    {
        for (size_t k = 0; k < out_dim; ++k)
        {
            float grad = 0.0f;
            for (size_t i = 0; i < batch; ++i)
            {
                grad += Input[i * in_dim + j] * Output_Loss[i * out_dim + k];
            }
            Weight[j * out_dim + k] -= lr * grad;
        }
    }
}

void relu_grad(const float* A, float* Grad, size_t batch, size_t out_dim)
{
    for (size_t i = 0; i < batch * out_dim; ++i)
    {
        Grad[i] = (A[i] > 0.0f) ? Grad[i] : 0.0f;
    }
}

float mean_acc(const unsigned char* result, const unsigned char* labels_array, size_t images_num, size_t num_classes)
{
    size_t correct = 0;
    for (size_t i = 0; i < images_num; ++i)
    {
        if (result[i] == labels_array[i])
        {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(images_num);
}

void argmax(const float* A, unsigned char* B, size_t num_classes, size_t images_num)
{
    for (size_t i = 0; i < images_num; ++i)
    {
        float max_val = A[i * num_classes];
        unsigned char max_idx = 0;
        for (size_t j = 1; j < num_classes; ++j)
        {
            if (A[i * num_classes + j] > max_val)
            {
                max_val = A[i * num_classes + j];
                max_idx = static_cast<unsigned char>(j);
            }
        }
        B[i] = max_idx;
    }
}
