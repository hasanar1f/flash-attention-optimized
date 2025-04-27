#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 32

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;  // over Br (Q rows)
    int ty = threadIdx.y;  // over Bc (K, V columns)
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S with rectangular tiles
    extern __shared__ float sram[];
    int tile_size_Qi = TILE_HEIGHT * d;
    int tile_size_KV = TILE_WIDTH * d;
    int tile_size_S  = TILE_HEIGHT * TILE_WIDTH;

    float* Qi = sram;
    float* Kj = &sram[tile_size_Qi];
    float* Vj = &sram[tile_size_Qi + tile_size_KV];
    float* S  = &sram[tile_size_Qi + 2 * tile_size_KV];

    for (int j = 0; j < Tc; j++) {
        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            if (ty < TILE_WIDTH) {
                Kj[(ty * d) + x] = K[qkv_offset + (j * TILE_WIDTH * d) + (ty * d) + x];
                Vj[(ty * d) + x] = V[qkv_offset + (j * TILE_WIDTH * d) + (ty * d) + x];
            }
        }
        __syncthreads();

        for (int i = 0; i < Tr; i++)  {
            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                if (tx < TILE_HEIGHT) {
                    Qi[(tx * d) + x] = Q[qkv_offset + (i * TILE_HEIGHT * d) + (tx * d) + x];
                }
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < TILE_WIDTH; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(TILE_WIDTH * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < TILE_WIDTH; y++) {
                S[(TILE_WIDTH * tx) + y] = __expf(S[(TILE_WIDTH * tx) + y] - row_m);
                row_l += S[(TILE_WIDTH * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < TILE_WIDTH; y++) {
                    pv += S[(TILE_WIDTH * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (i * TILE_HEIGHT * d) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (i * TILE_HEIGHT * d) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    const int Bc = TILE_WIDTH;  // 64
    const int Br = TILE_HEIGHT; // 128

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    const int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);
    dim3 block_dim(Br, Bc);

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}
