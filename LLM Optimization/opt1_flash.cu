#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

/*
Optimization 1: Blocked Matrix Multiplication

Enhances data locality and cache utilization by dividing matrices into smaller blocks, optimizing the use of cache memory.

Self CPU time total: 10.257ms
Self CUDA time total: 6.033ms

*/

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
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

    for (int jj = 0; jj < Tc; jj += Bc) {
        for (int j = 0; j < Bc && jj + j < Tc; j++) {
            for (int x = 0; x < d; x++) {
                Kj[(tx * d) + x] = K[qkv_offset + (tile_size * (jj + j)) + (tx * d) + x];
                Vj[(tx * d) + x] = V[qkv_offset + (tile_size * (jj + j)) + (tx * d) + x];
            }
            __syncthreads();

            for (int ii = 0; ii < Tr; ii += Br) {
                for (int i = 0; i < Br && ii + i < Tr; i++) {
                    for (int x = 0; x < d; x++) {
                        Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * (ii + i)) + (tx * d) + x];
                    }
                    float row_m_prev = m[lm_offset + (Br * (ii + i)) + tx];
                    float row_l_prev = l[lm_offset + (Br * (ii + i)) + tx];

                    float row_m = -INFINITY;
                    for (int y = 0; y < Bc; y++) {
                        float sum = 0;
                        for (int x = 0; x < d; x++) {
                            sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                        }
                        sum *= softmax_scale;
                        S[(Bc * tx) + y] = sum;

                        if (sum > row_m)
                            row_m = sum;
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
                        O[qkv_offset + (tile_size * (ii + i)) + (tx * d) + x] = (1 / row_l_new) \
                            * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * (ii + i)) + (tx * d) + x]) \
                            + (__expf(row_m - row_m_new) * pv));
                    }
                    m[lm_offset + (Br * (ii + i)) + tx] = row_m_new;
                    l[lm_offset + (Br * (ii + i)) + tx] = row_l_new;
                }
                __syncthreads();
            }
        }
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Bc);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}