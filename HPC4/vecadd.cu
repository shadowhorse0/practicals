#include<bits/stdc++.h>
using namespace std;
// CUDA kernel for vector addition
__global__
void vectorAddition(const int* A, const int* B, int* C, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        C[index] = A[index] + B[index];
    }
}

// Function to perform vector addition sequentially
void sequentialVectorAddition(const int* A, const int* B, int* C, int size)
{
    for (int i = 0; i < size; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int size = 1000000;  // Size of the vectors

    // Allocate memory for vectors on host
    int* A = new int[size];
    int* B = new int[size];
    int* C = new int[size];

    // Initialize vectors with random values
    for (int i = 0; i < size; ++i)
    {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate memory for vectors on device
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, sizeof(int) * size);
    cudaMalloc((void**)&d_B, sizeof(int) * size);
    cudaMalloc((void**)&d_C, sizeof(int) * size);

    // Copy input vectors from host to device
    cudaMemcpy(d_A, A, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int) * size, cudaMemcpyHostToDevice);

    // Set up thread configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    // Start timer for parallel algorithm
    auto startParallel = chrono::high_resolution_clock::now();

    // Launch kernel for vector addition in parallel
    vectorAddition<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    // Wait for kernel to finish execution
    cudaDeviceSynchronize();

    // End timer for parallel algorithm
    auto endParallel = chrono::high_resolution_clock::now();

    // Copy result vector from device to host
    cudaMemcpy(C, d_C, sizeof(int) * size, cudaMemcpyDeviceToHost);

    // Start timer for sequential algorithm
    auto startSequential = chrono::high_resolution_clock::now();

    // Perform vector addition sequentially
    sequentialVectorAddition(A, B, C, size);

    // End timer for sequential algorithm
    auto endSequential = chrono::high_resolution_clock::now();

    // Calculate elapsed time for parallel algorithm
    auto durationParallel = chrono::duration_cast<chrono::microseconds>(endParallel - startParallel);

    // Calculate elapsed time for sequential algorithm
    auto durationSequential = chrono::duration_cast<chrono::microseconds>(endSequential - startSequential);

    // Print performance results
    cout << "Parallel Algorithm Time: " << durationParallel.count() << " microseconds" << endl;
    cout << "Sequential Algorithm Time: " << durationSequential.count() << " microseconds" << endl;

    // Print the results of vector addition (first 10 elements)
    scout << "Vector Addition Result:" << endl;
    for (int i = 0; i < 10; ++i)
    {
        scout << A[i] << " + " << B[i] << " = " << C[i] << endl;
    }

    // Free memory
    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
