#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {

__global__ void loop_unroll_inference(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float sum = 0.0f;

        // 使用 loop unrolling 优化计算
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
            if (idx + i < n) { // 边界检查
                sum += input[idx + i];
            }
        }
        
        output[idx] = sum;
    }
}

void launch_loop_unroll_inference(const float* input, float* output, int n, int threads_per_block) {
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    loop_unroll_inference<<<blocks, threads_per_block>>>(input, output, n);
}

}
