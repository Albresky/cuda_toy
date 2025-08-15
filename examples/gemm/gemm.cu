/*
 * @Author: Albresky albre02@outlook.com
 * @Date: 2025-08-15 12:08:34
 * @LastEditors: Albresky albre02@outlook.com
 * @LastEditTime: 2025-08-15 17:24:54
 * @FilePath: /cuda/toy/gemm/baseline.cu
 * @Description: GEMM 优化
 */

#include "Helper.cuh"
#include <cmath> // For fabs and fmax
#include <cuda_runtime.h>
#include <iostream>

#define OP2 // Optimize Level
#define IDX2C(r, c, ld) ((r) * (ld) + (c))

struct Matrix {
  int M;
  int K;
  int N;
  float *elements;

public:
  Matrix(int m = 0, int k = 0, int n = 0) : M(m), K(k), N(n) {
    elements = nullptr;
  }
};

// Level 0: 基线 Kernel 实现
__global__ void mmKnlL0(const Matrix A, const Matrix B, Matrix C) {
  // 计算当前线程应该处理的C矩阵的全局行和列
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界检查，防止处理到矩阵外的元素
  if (row < C.M && col < C.N) {
    float Cvalue = 0.0f;
    for (int k = 0; k < A.K; ++k) {
      Cvalue += A.elements[row * A.K + k] * B.elements[k * B.N + col];
    }
    C.elements[row * C.N + col] = Cvalue;
  }
}

// Kernel for Level 1: Tiling with Shared Memory
#define BLOCK_SIZE 32
__global__ void mmKnlL1(const Matrix A, const Matrix B, Matrix C) {
  // 声明共享内存来存储A和B的Tile
  // __shared__关键字表明这个变量存在于共享内存中
  __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

  // C的行和列，由线程的全局ID决定 (与Level 0相同)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 线程在Block内的局部ID
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float Cvalue = 0.0f;

  // Tile的数量
  int numTiles = (A.K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // 遍历所有的Tile
  // tidx.x 和 tidx.y 分别是块内的线程索引
  for (int i = 0; i < numTiles; ++i) {
    // --- 协作加载数据到共享内存 ---
    // 每个线程负责从全局内存加载一个A的元素和一个B的元素到共享内存

    int aCol = i * BLOCK_SIZE + tx;
    int bRow = i * BLOCK_SIZE + ty;

    // 条件表达式：确保加载的全局内存地址不越界
    sA[ty][tx] = row < A.M && aCol < A.K ? A.elements[row * A.K + aCol] : 0.0f;
    sB[ty][tx] = bRow < B.K && col < B.N ? B.elements[bRow * B.N + col] : 0.0f;

    // --- 块内同步 ---
    // 必须等待所有线程都将数据加载到共享内存后，才能开始计算
    // 否则，可能会有线程使用到尚未被加载的脏数据
    __syncthreads();

    // --- 从共享内存计算部分点积 ---
    // Block内的每个线程都执行这部分计算
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Cvalue += sA[ty][k] * sB[k][tx];
    }

    // --- 再次同步 ---
    // 确保当前Tile的计算全部完成后，才能进入下一次循环去加载新的Tile
    // 否则，可能会有线程提前执行，覆盖掉其他线程还在使用的共享内存数据
    __syncthreads();
  }

  // 将最终结果写回全局内存
  if (row < C.M && col < C.N) {
    C.elements[row * C.N + col] = Cvalue;
  }
}

// Kernel for Level 2: More computation per thread
#define TILE_DIM 32                         // Tile dimension
#define THREAD_ROWS 8                       // 每个线程计算 C 的8行
#define THREAD_COLS 4                       // 每个线程计算 C 的4列
#define BLOCK_ROWS (TILE_DIM / THREAD_ROWS) // 32/8=4
#define BLOCK_COLS (TILE_DIM / THREAD_COLS) // 32/4=8

__global__ void mmKnlL2(const Matrix A, const Matrix B, Matrix C) {
  // Shared memory for tiles of A and B
  __shared__ float sA[TILE_DIM][TILE_DIM];
  __shared__ float sB[TILE_DIM][TILE_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 定位线程块处理的C矩阵的左上角
  int tile_row = blockIdx.y * TILE_DIM;
  int tile_col = blockIdx.x * TILE_DIM;

  // 定位线程处理的C矩阵的左上角
  // 每个线程现在负责BLOCK_ROWS中的一行
  int thd_row_begin = tile_row + ty * THREAD_ROWS;
  int thd_col_begin = tile_col + tx * THREAD_COLS;

  // 用于存储结果的寄存器数组
  float Creg[THREAD_ROWS][THREAD_COLS] = {{0.0f}};

  int numTiles = (A.K + TILE_DIM - 1) / TILE_DIM;

  // Loop over the tiles of A and B
  for (int i = 0; i < numTiles; ++i) {
    // --- Load tiles into shared memory ---
    // 32个线程(4x8)协作加载32x32的tile, 每个线程加载32个元素
    // 这里采用每个线程加载一个8x4的区域

    // Load Tile A
    for (int k = tx; k < TILE_DIM; k += blockDim.x)
      for (int j = 0; j < THREAD_ROWS; ++j) {
        int sA_m = ty * THREAD_ROWS + j;
        int gA_m = tile_row + sA_m;
        int gA_k = i * TILE_DIM + k;
        sA[sA_m][k] =
            (gA_m < A.M && gA_k < A.K) ? A.elements[gA_m * A.K + gA_k] : 0.0f;
      }

    // Load Tile B
    for (int k = ty; k < TILE_DIM; k += blockDim.y)
      for (int j = 0; j < THREAD_COLS; ++j) {
        int sB_n = tx * THREAD_COLS + j;
        int gB_k = i * TILE_DIM + k;
        int gB_n = tile_col + sB_n;
        sB[k][tx * THREAD_COLS + j] =
            (gB_k < B.K && gB_n < B.N) ? B.elements[gB_k * B.N + gB_n] : 0.0f;
      }
    __syncthreads();

    // --- Compute partial result from shared memory ---
    for (int k = 0; k < TILE_DIM; ++k) {
      for (int m = 0; m < THREAD_ROWS; ++m) {
        for (int n = 0; n < THREAD_COLS; ++n) {
          int sA_m = ty * THREAD_ROWS + m;
          int sB_n = tx * THREAD_COLS + n;
          Creg[m][n] += sA[sA_m][k] * sB[k][sB_n];
        }
      }
    }
    __syncthreads();
  }
  __syncthreads();

  // --- Write result back to global memory ---
  for (int m = 0; m < THREAD_ROWS; ++m) {
    for (int n = 0; n < THREAD_COLS; ++n) {
      int c_m = thd_row_begin + m;
      int c_n = thd_col_begin + n;
      if (c_m < C.M && c_n < C.N) {
        C.elements[c_m * C.N + c_n] = Creg[m][n];
      }
    }
  }
}

int main() {
  // --- 1. 初始化 ---
  const int ITER = 1000;

  // 1024x1024 矩阵
  const int N = 1 << 10;

  Matrix h_A(N, N, N);
  Matrix h_B(N, N, N);
  Matrix h_C(N, N, N);

  size_t aBytes = (size_t)h_A.M * h_A.K * sizeof(float); // M*K
  size_t bBytes = (size_t)h_B.K * h_B.N * sizeof(float); // K*N
  size_t cBytes = (size_t)h_C.M * h_C.N * sizeof(float); // M*N

  // 在Device上分配矩阵数据内存
  h_A.elements = (float *)malloc(aBytes);
  h_B.elements = (float *)malloc(bBytes);
  h_C.elements = (float *)malloc(cBytes);

  // 在Host上创建临时数据并初始化

  for (int i = 0; i < h_A.M * h_A.K; ++i)
    h_A.elements[i] = 1.0f;
  for (int i = 0; i < h_B.K * h_B.N; ++i)
    h_B.elements[i] = 2.0f;

  Matrix d_A(h_A), d_B(h_B), d_C(h_C);

  // 为device矩阵分配内存
  CUDA_CHECK(cudaMalloc(&d_A.elements, aBytes));
  CUDA_CHECK(cudaMalloc(&d_B.elements, bBytes));
  CUDA_CHECK(cudaMalloc(&d_C.elements, cBytes));

  // 将Host数据拷贝到Device
  CUDA_CHECK(
      cudaMemcpy(d_A.elements, h_A.elements, aBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B.elements, h_B.elements, bBytes, cudaMemcpyHostToDevice));

  // --- 2. Kernel执行与计时 ---
  dim3 blockSize;
  dim3 gridSize;

#if defined(OP0) || defined(OP1)
  blockSize = dim3(TILE_DIM, TILE_DIM);
  gridSize = dim3((h_C.N + blockSize.x - 1) / blockSize.x,
                  (h_C.M + blockSize.y - 1) / blockSize.y);
#elif defined(OP2)
  blockSize = dim3(BLOCK_COLS, BLOCK_ROWS); // 8, 4
  gridSize = dim3((h_C.N + TILE_DIM - 1) / TILE_DIM,
                  (h_C.M + TILE_DIM - 1) / TILE_DIM);
#endif
  // 创建CUDA事件用于计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 记录开始时间
  CUDA_CHECK(cudaEventRecord(start));

  // 执行Kernel
  for (int i = 0; i < ITER; ++i) {
#if defined(OP0)
    mmKnlL0<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP1)
    mmKnlL1<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP2)
    mmKnlL2<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#else
    std::cout << "[Error] Optimize Level not targeted or defined." << std::endl;
    exit(0);
#endif
  }
  // 记录结束时间
  CUDA_CHECK(cudaEventRecord(stop));

  // 等待Kernel执行完毕
  CUDA_CHECK(cudaEventSynchronize(stop));

  // 计算执行时间
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Total iteration:" << ITER << "\n"
            << "Tatal Kernel Execution Time: " << milliseconds << " ms\n"
            << "Average Kernel Execution Time: " << milliseconds / ITER
            << " ms\n"
            << std::endl;

  // --- 3. 结果验证 ---
  CUDA_CHECK(
      cudaMemcpy(h_C.elements, d_C.elements, cBytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  float maxError = 0.0f;
  for (int i = 0; i < N * N; ++i) {
    float fab = fabs(h_C.elements[i] - (2.0f * N));
    if (fab > 0.00001)
      ++errors;
    maxError = fmax(maxError, fab);
  }
  if (!errors)
    std::cout << "Test PASS!" << std::endl;
  else
    std::cout << "Errors:" << errors << ", Max Error: " << maxError
              << std::endl;

  // --- 4. 资源释放 ---
  free(h_A.elements);
  free(h_B.elements);
  free(h_C.elements);
  CUDA_CHECK(cudaFree(d_A.elements));
  CUDA_CHECK(cudaFree(d_B.elements));
  CUDA_CHECK(cudaFree(d_C.elements));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}