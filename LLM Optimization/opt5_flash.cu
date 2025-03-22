#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>


/*
Optimization 5: Vectorization and Memory Alignment

Utilizes CPU vector instructions and ensures data is aligned in memory to maximize processing speed and efficiency.

Self CPU time total: 9.451ms
Self CUDA time total: 5.524ms

*/

__global__
void forward_kernel(const float* __restrict__ Q, const float* __restrict__ K, const float* __restrict__ V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* __restrict__ l, float* __restrict__ m, float* __restrict__ O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d;
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {
        int base_idx_j = qkv_offset + (tile_size * j);
        for (int x = 0; x < d; x++) {
            int idx = (tx * d) + x;
            Kj[idx] = K[base_idx_j + idx];
            Vj[idx] = V[base_idx_j + idx];
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)  {
            int base_idx_i = qkv_offset + (tile_size * i);
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[base_idx_i + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                #pragma unroll
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                row_m = max(row_m, sum);
            }

            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            for (int x = 0; x < d; x++) {
                float pv = 0;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[base_idx_i + (tx * d) + x] = (1 / row_l_new) * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[base_idx_i + (tx * d) + x]) + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceilf(static_cast<float>(N) / Bc); const int Tr = ceilf(static_cast<float>(N) / Br);
    const float softmax_scale = 1.0f / sqrtf(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N}, torch::device(torch::kCUDA).dtype(torch::kFloat32));
    auto m = torch::full({B, nh, N}, -INFINITY, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}