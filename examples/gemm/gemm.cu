/*
 * @Author: Albresky albre02@outlook.com
 * @Date: 2025-08-15 12:08:34
 * @LastEditors: Albresky albre02@outlook.com
 * @LastEditTime: 2025-08-15 21:34:33
 * @FilePath: /cuda/toy/gemm/baseline.cu
 * @Description: GEMM 优化
 */

#include "Helper.cuh"
#include "gemm.cuh"
#include <cmath> // For fabs and fmax
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_fp16.h> // For __half data type
#include <iostream>
#include <mma.h> // For WMMA API

// Level 0: 基线 Kernel 实现
/**
 * Note: 每个 thread 单独计算一个结果元素 C [m][n]
 */
__global__ void mmKnlL0(const Matrix A, const Matrix B, Matrix C) {
  // 计算当前线程应该处理的 C 矩阵的全局行和列
  int row = blockIdx.y * /* Block in Grid */ blockDim.y +
            /* Thread in block */ threadIdx.y;
  int col = blockIdx.x * /* Block in Grid */ blockDim.x +
            /* Thread in block */ threadIdx.x;

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
/**
 * Note: 将结果矩阵分块 tiling，每个线程块 block 计算结果矩阵的一个 tile
 * 每个 thread 还是只算 1 个结果元素 C [m][n]
 */
#define BLOCK_SIZE 32
#define TILE_DIM 32 // Tile dimension
__global__ void mmKnlL1(const Matrix A, const Matrix B, Matrix C) {
  // 声明共享内存来存储 A 和 B 的 Tile
  //__shared__关键字表明这个变量存在于共享内存中
  // (线程块内的共享内存 Shared Memory)
  __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

  // C 的行和列，由线程的全局 ID 决定 (与 Level 0 相同)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // 线程在 Block 内的局部 ID
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  float Cvalue = 0.0f;

  // Tile 的数量
  int numTiles = (A.K + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // 遍历所有的 Tile
  // tx 和 ty 分别是块内的线程索引
  for (int i = 0; i < numTiles; ++i) {
    //--- 协作加载数据到共享内存 ---
    // 每个线程负责从全局内存加载一个 A 的元素和一个 B 的元素到共享内存

    int aCol = i * BLOCK_SIZE + tx;
    int bRow = i * BLOCK_SIZE + ty;

    // 条件表达式：确保加载的全局内存地址不越界
    sA[ty][tx] = row < A.M && aCol < A.K ? A.elements[row * A.K + aCol] : 0.0f;
    sB[ty][tx] = bRow < B.K && col < B.N ? B.elements[bRow * B.N + col] : 0.0f;

    //--- 块内同步 ---
    // 必须等待所有线程都将数据加载到共享内存后，才能开始计算
    // 否则，可能会有线程使用到尚未被加载的脏数据
    __syncthreads();

    //--- 从共享内存计算部分点积 ---
    // Block 内的每个线程都执行这部分计算
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      Cvalue += sA[ty][k] * sB[k][tx];
    }

    //--- 再次同步 ---
    // 确保当前 Tile 的计算全部完成后，才能进入下一次循环去加载新的 Tile
    // 否则，可能会有线程提前执行，覆盖掉其他线程还在使用的共享内存数据
    __syncthreads();
  }

  // 将最终结果写回全局内存
  if (row < C.M && col < C.N) {
    C.elements[row * C.N + col] = Cvalue;
  }
}

// Kernel for Level 2: More computation per thread
/**
 * Note: L1 增强版。将结果矩阵分块 tiling，每个线程块 block
 * 计算结果矩阵的一个大 tile（粗粒度）， 每个 thread 计算 1
 * 个小 tile（8*4，细粒度）、对应结果元素 C [m-7:m][n-3:n]
 */
#define THREAD_ROWS 8                       // 每个线程计算 C 的 8 行
#define THREAD_COLS 4                       // 每个线程计算 C 的 4 列
#define BLOCK_ROWS (TILE_DIM / THREAD_ROWS) // 32/8=4
#define BLOCK_COLS (TILE_DIM / THREAD_COLS) // 32/4=8

__global__ void mmKnlL2(const Matrix A, const Matrix B, Matrix C) {
  // Shared memory for tiles of A and B
  __shared__ float sA[TILE_DIM][TILE_DIM];
  __shared__ float sB[TILE_DIM][TILE_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // 定位线程块处理的 C 矩阵的左上角
  int tile_row = blockIdx.y * TILE_DIM;
  int tile_col = blockIdx.x * TILE_DIM;

  // 定位线程处理的 C 矩阵的左上角
  // 每个线程现在负责 BLOCK_ROWS 中的一行
  int thd_row_begin = tile_row + ty * THREAD_ROWS;
  int thd_col_begin = tile_col + tx * THREAD_COLS;

  // 用于存储结果的寄存器数组
  float Creg[THREAD_ROWS][THREAD_COLS] = {{0.0f}};

  int numTiles = (A.K + TILE_DIM - 1) / TILE_DIM;

  // Loop over the tiles of A and B
  for (int i = 0; i < numTiles; ++i) {
    //--- Load tiles into shared memory ---
    // 32 个线程 (4x8) 协作加载 32x32 的 tile, 每个线程加载 32 个元素
    // 这里采用每个线程加载一个 8x4 的区域

    // Load Tile A
    // 1 Tile: 32x32
    for (int k = tx; k < TILE_DIM; k += blockDim.x)
      // 1 thread --> THREAD_ROWS
      for (int j = 0; j < THREAD_ROWS; ++j) {
        int sA_m = ty * THREAD_ROWS + j;
        int gA_m = tile_row + sA_m;
        int gA_k = i * TILE_DIM + k;
        sA[sA_m][k] =
            (gA_m < A.M && gA_k < A.K) ? A.elements[gA_m * A.K + gA_k] : 0.0f;
      }

    // Load Tile B
    for (int k = ty; k < TILE_DIM; k += blockDim.y)
      // 1 thread --> THREAD_COLS
      for (int j = 0; j < THREAD_COLS; ++j) {
        int sB_n = tx * THREAD_COLS + j;
        int gB_k = i * TILE_DIM + k;
        int gB_n = tile_col + sB_n;
        sB[k][tx * THREAD_COLS + j] =
            (gB_k < B.K && gB_n < B.N) ? B.elements[gB_k * B.N + gB_n] : 0.0f;
      }
    __syncthreads();

    //--- Compute partial result from shared memory ---
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

  //--- Write result back to global memory ---
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

// Kernel for Level 3: Double Buffering in Shared Memory with Sync Pipeline
/**
 * Note: 基于L2，用双缓冲实现数据存取、计算，有效 pipeline 指令
 */
__global__ void mmKnlL3(const Matrix A, const Matrix B, Matrix C) {
  // 将共享内存大小加倍，实现双缓冲
  __shared__ float sA[2][TILE_DIM][TILE_DIM];
  __shared__ float sB[2][TILE_DIM][TILE_DIM];

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tile_row = blockIdx.y * TILE_DIM;
  int tile_col = blockIdx.x * TILE_DIM;

  int thd_row_begin = tile_row + ty * THREAD_ROWS;
  int thd_col_begin = tile_col + tx * THREAD_COLS;

  float Creg[THREAD_ROWS][THREAD_COLS] = {{0.0f}};

  int numTiles = (A.K + TILE_DIM - 1) / TILE_DIM;

  //--- 预加载：在主循环开始前，提前加载第一个 Tile 到 Buffer 0 ---
  int i = 0;
  // 加载逻辑与 L2 完全相同，只是写入 sA [0] 和 sB [0]
  // Load Tile A into sA [0]
  for (int k = tx; k < TILE_DIM; k += blockDim.x)
    for (int j = 0; j < THREAD_ROWS; ++j) {
      int sA_m = ty * THREAD_ROWS + j;
      int gA_m = tile_row + sA_m;
      int gA_k = i * TILE_DIM + k;
      sA[0][sA_m][k] =
          (gA_m < A.M && gA_k < A.K) ? A.elements[gA_m * A.K + gA_k] : 0.0f;
    }
  // Load Tile B into sB [0]
  for (int k = ty; k < TILE_DIM; k += blockDim.y)
    for (int j = 0; j < THREAD_COLS; ++j) {
      int sB_n = tx * THREAD_COLS + j;
      int gB_k = i * TILE_DIM + k;
      int gB_n = tile_col + sB_n;
      sB[0][k][sB_n] =
          (gB_k < B.K && gB_n < B.N) ? B.elements[gB_k * B.N + gB_n] : 0.0f;
    }
  __syncthreads();

  //--- 主循环：计算和加载重叠进行 ---
  for (i = 0; i < numTiles - 1; ++i) {
    // 定义当前 buffer 和下一个 buffer 的索引
    int current_buf = i % 2;
    int next_buf = (i + 1) % 2;

    // 异步加载下一个 Tile 到 next_buf
    // Load Tile A into sA [next_buf]
    for (int k = tx; k < TILE_DIM; k += blockDim.x)
      for (int j = 0; j < THREAD_ROWS; ++j) {
        int sA_m = ty * THREAD_ROWS + j;
        int gA_m = tile_row + sA_m;
        int gA_k = (i + 1) * TILE_DIM + k;
        sA[next_buf][sA_m][k] =
            (gA_m < A.M && gA_k < A.K) ? A.elements[gA_m * A.K + gA_k] : 0.0f;
      }
    // Load Tile B into sB [next_buf]
    for (int k = ty; k < TILE_DIM; k += blockDim.y)
      for (int j = 0; j < THREAD_COLS; ++j) {
        int sB_n = tx * THREAD_COLS + j;
        int gB_k = (i + 1) * TILE_DIM + k;
        int gB_n = tile_col + sB_n;
        sB[next_buf][k][sB_n] =
            (gB_k < B.K && gB_n < B.N) ? B.elements[gB_k * B.N + gB_n] : 0.0f;
      }

    // 使用 current_buf 中的数据进行计算
    for (int k = 0; k < TILE_DIM; ++k) {
      for (int m = 0; m < THREAD_ROWS; ++m) {
        for (int n = 0; n < THREAD_COLS; ++n) {
          int sA_m = ty * THREAD_ROWS + m;
          int sB_n = tx * THREAD_COLS + n;
          Creg[m][n] += sA[current_buf][sA_m][k] * sB[current_buf][k][sB_n];
        }
      }
    }
    __syncthreads(); // 确保下一轮计算开始前，上一轮的加载已全部完成
  }

  //--- 处理最后一个 Tile 的计算 ---
  // 因为循环只到 numTiles-1，最后一个 tile 的数据已加载，但尚未计算
  int last_buf = (numTiles - 1) % 2;
  for (int k = 0; k < TILE_DIM; ++k) {
    for (int m = 0; m < THREAD_ROWS; ++m) {
      for (int n = 0; n < THREAD_COLS; ++n) {
        int sA_m = ty * THREAD_ROWS + m;
        int sB_n = tx * THREAD_COLS + n;
        Creg[m][n] += sA[last_buf][sA_m][k] * sB[last_buf][k][sB_n];
      }
    }
  }

  //--- 写回结果 (与 L2 相同) ---
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

// Kernel for Level 4: Double Buffering with Asynchronous Pipeline in Block
// level NOTE: Requires Compute Capability 9.0+ (Hopper Architecture or newer)
__global__ void mmKnlL4(const Matrix A, const Matrix B, Matrix C) {
  // --- 1. 初始化 ---
  // 共享内存双缓冲
  __shared__ float sA[2][TILE_DIM][TILE_DIM];
  __shared__ float sB[2][TILE_DIM][TILE_DIM];

  // 创建 pipeline 对象，由整个 Block 共享和同步
  // cuda::thread_scope_block 表示这个 pipeline 由整个 Block 共享和同步。
  // shared pipeline state for block-scoped pipeline (StagesCount 2 is common)
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss;

  // get block group
  auto group = cooperative_groups::this_thread_block();

  // make a block-scoped pipeline (must be called by every thread in group)
  cuda::pipeline<cuda::thread_scope_block> pipe =
      cuda::make_pipeline(group, &pss);

  // 线程和块索引 (与 L2/L3 相同)
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tile_row = blockIdx.y * TILE_DIM;
  const int tile_col = blockIdx.x * TILE_DIM;
  const int thd_row_begin = tile_row + ty * THREAD_ROWS;
  const int thd_col_begin = tile_col + tx * THREAD_COLS;

  // 寄存器数组存储结果 (与 L2/L3 相同)
  float Creg[THREAD_ROWS][THREAD_COLS] = {{0.0f}};

  int numTiles = (A.K + TILE_DIM - 1) / TILE_DIM;

  // 异步加载Tile A的函数
  // 整个Block协作，每个线程负责拷贝一行
  // ---- 工具函数：行拷贝 + 边界零填充 ----
  auto copy_row_async_pad = [&](float *dst_row, const float *src_row,
                                int valid_elems) {
    if (valid_elems == TILE_DIM) {
      // 满行：直接异步 memcpy
      cuda::memcpy_async(dst_row, src_row, TILE_DIM * sizeof(float), pipe);
    } else if (valid_elems > 0) {
      // 部分有效：先拷贝有效段，再零填充余下
      cuda::memcpy_async(dst_row, src_row, valid_elems * sizeof(float), pipe);
      // 零填充（同步写共享内存，不走 pipeline；零填充很快）
      for (int j = valid_elems; j < TILE_DIM; ++j)
        dst_row[j] = 0.0f;
    } else {
      // 全越界：整行置零
      for (int j = 0; j < TILE_DIM; ++j)
        dst_row[j] = 0.0f;
    }
  };
  __syncthreads();

  // --- 2. 启动流水线：预加载第一个 Tile (Tile 0) ---
  // 这是流水线的第一步，"prime the pump"
  {
    pipe.producer_acquire();

    // A: tile 行拷贝（行内按 K 连续）
    // 将 block 中所有线程映射为 0..(TILE_DIM-1) 的行索引（stride 方式）
    const int tlinear = ty * blockDim.x + tx;
    for (int r = tlinear; r < TILE_DIM; r += blockDim.x * blockDim.y) {
      const int gA_m = tile_row + r;
      const int gA_k0 = 0;
      float *dst = &sA[0][r][0];
      if (gA_m < A.M) {
        const int valid = max(0, min(TILE_DIM, A.K - gA_k0));
        const float *src = &A.elements[gA_m * A.K + gA_k0];
        copy_row_async_pad(dst, src, valid);
      } else {
        for (int j = 0; j < TILE_DIM; ++j)
          dst[j] = 0.0f;
      }
    }

    // B: tile 行拷贝（每行固定 k，列方向 N 连续）
    for (int r = tlinear; r < TILE_DIM; r += blockDim.x * blockDim.y) {
      const int gB_k = 0 + r;
      float *dst = &sB[0][r][0];
      if (gB_k < B.K) {
        const int valid = max(0, min(TILE_DIM, B.N - tile_col));
        const float *src = &B.elements[gB_k * B.N + tile_col];
        copy_row_async_pad(dst, src, valid);
      } else {
        for (int j = 0; j < TILE_DIM; ++j)
          dst[j] = 0.0f;
      }
    }

    pipe.producer_commit();
  }

  // --- 3. 主循环：计算和加载流水线化 ---
  // 循环比 L2/L3 少一次，因为最后一个 tile 在循环后单独处理
  for (int i = 0; i < numTiles - 1; ++i) {
    // 定义当前计算用的 buffer 和下一个加载用的 buffer
    const int compute_buf = i % 2;
    const int load_buf = (i + 1) % 2;
    const int gk0_next = (i + 1) * TILE_DIM;

    // --- 异步加载下一个 Tile (Tile i+1) ---
    pipe.producer_acquire();

    // --- 使用当前 Tile 的数据进行计算 ---
    // 此时，sA[compute_buf] 和 sB[compute_buf] 的数据已保证就绪
    // load A
    {
      const int tlinear = ty * blockDim.x + tx;
      for (int r = tlinear; r < TILE_DIM; r += blockDim.x * blockDim.y) {
        const int gA_m = tile_row + r;
        float *dst = &sA[load_buf][r][0];
        if (gA_m < A.M) {
          const int valid = max(0, min(TILE_DIM, A.K - gk0_next));
          const float *src = &A.elements[gA_m * A.K + gk0_next];
          copy_row_async_pad(dst, src, valid);
        } else {
          for (int j = 0; j < TILE_DIM; ++j)
            dst[j] = 0.0f;
        }
      }
    }
    // B
    {
      const int tlinear = ty * blockDim.x + tx;
      for (int r = tlinear; r < TILE_DIM; r += blockDim.x * blockDim.y) {
        const int gB_k = gk0_next + r;
        float *dst = &sB[load_buf][r][0];
        if (gB_k < B.K) {
          const int valid = max(0, min(TILE_DIM, B.N - tile_col));
          const float *src = &B.elements[gB_k * B.N + tile_col];
          copy_row_async_pad(dst, src, valid);
        } else {
          for (int j = 0; j < TILE_DIM; ++j)
            dst[j] = 0.0f;
        }
      }
    }
    pipe.producer_commit();

    // 等待当前 compute_buf 数据可用，然后计算
    pipe.consumer_wait();
    __syncthreads();

    // 使用 current_buf 中的数据进行计算
    for (int k = 0; k < TILE_DIM; ++k) {
      for (int m = 0; m < THREAD_ROWS; ++m) {
        for (int n = 0; n < THREAD_COLS; ++n) {
          int sA_m = ty * THREAD_ROWS + m;
          int sB_n = tx * THREAD_COLS + n;
          Creg[m][n] += sA[compute_buf][sA_m][k] * sB[compute_buf][k][sB_n];
        }
      }
    }

    // --- 提交下一轮的拷贝任务 ---
    pipe.consumer_release();

    // 确保手动加载B的部分对所有线程可见
    __syncthreads();
  }

  // --- 4. 排空流水线：处理最后一个 Tile (Tile numTiles-1) 的计算 ---
  {
    const int last_buf = (numTiles - 1) % 2;
    pipe.consumer_wait();
    __syncthreads();

    for (int k = 0; k < TILE_DIM; ++k) {
      for (int m = 0; m < THREAD_ROWS; ++m) {
        for (int n = 0; n < THREAD_COLS; ++n) {
          const int sA_m = ty * THREAD_ROWS + m;
          const int sB_n = tx * THREAD_COLS + n;
          Creg[m][n] += sA[last_buf][sA_m][k] * sB[last_buf][k][sB_n];
        }
      }
    }
    pipe.consumer_release();
    __syncthreads();
  }

  // --- 5. 写回结果 (与 L2 完全相同) ---
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

// Kernel for Level 5: Manual Loop Unrolling
__global__ void mmKnlL5(const Matrix A, const Matrix B, Matrix C) {
  // --- 声明、索引计算、数据加载部分与L2完全相同 ---
  __shared__ float sA[TILE_DIM][TILE_DIM];
  __shared__ float sB[TILE_DIM][TILE_DIM];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tile_row = blockIdx.y * TILE_DIM;
  int tile_col = blockIdx.x * TILE_DIM;
  int thd_row_begin = tile_row + ty * THREAD_ROWS;
  int thd_col_begin = tile_col + tx * THREAD_COLS;
  float Creg[THREAD_ROWS][THREAD_COLS] = {{0.0f}};
  int numTiles = (A.K + TILE_DIM - 1) / TILE_DIM;

  for (int i = 0; i < numTiles; ++i) {
    // --- 加载逻辑与L2完全相同 ---
    for (int k = tx; k < TILE_DIM; k += blockDim.x)
      for (int j = 0; j < THREAD_ROWS; ++j) {
        int sA_m = ty * THREAD_ROWS + j;
        int gA_m = tile_row + sA_m;
        int gA_k = i * TILE_DIM + k;
        sA[sA_m][k] =
            (gA_m < A.M && gA_k < A.K) ? A.elements[gA_m * A.K + gA_k] : 0.0f;
      }
    for (int k = ty; k < TILE_DIM; k += blockDim.y)
      for (int j = 0; j < THREAD_COLS; ++j) {
        int sB_n = tx * THREAD_COLS + j;
        int gB_k = i * TILE_DIM + k;
        int gB_n = tile_col + sB_n;
        sB[k][sB_n] =
            (gB_k < B.K && gB_n < B.N) ? B.elements[gB_k * B.N + gB_n] : 0.0f;
      }
    __syncthreads();

    // --- 【L4 核心改动】手动展开内层计算循环 ---
    const int UNROLL_FACTOR = 4; // 展开因子
    for (int k = 0; k < TILE_DIM; k += UNROLL_FACTOR) {
      for (int m = 0; m < THREAD_ROWS; ++m) {
        for (int n = 0; n < THREAD_COLS; ++n) {
          int sA_m = ty * THREAD_ROWS + m;
          int sB_n = tx * THREAD_COLS + n;

          // 展开4次
          Creg[m][n] += sA[sA_m][k + 0] * sB[k + 0][sB_n];
          Creg[m][n] += sA[sA_m][k + 1] * sB[k + 1][sB_n];
          Creg[m][n] += sA[sA_m][k + 2] * sB[k + 2][sB_n];
          Creg[m][n] += sA[sA_m][k + 3] * sB[k + 3][sB_n];
        }
      }
    }
    __syncthreads();
  }

  // --- 写回结果 (与L2完全相同) ---
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

// --- Level 6: 使用Tensor Cores (WMMA) ---
// 定义WMMA操作的矩阵块尺寸
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Block tile size. Must be a multiple of WMMA shapes.
// Let's define a 64x64 tile per thread block.
#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define BLOCK_TILE_K 16

// Warps per block
#define WARPS_PER_BLOCK_M (BLOCK_TILE_M / WMMA_M) // 64/16 = 4
#define WARPS_PER_BLOCK_N (BLOCK_TILE_N / WMMA_N) // 64/16 = 4

__global__ void mmKnlL6(const Matrix A, const Matrix B, Matrix C) {
  using namespace nvcuda::wmma;

  // --- 线程块和 Warp 索引计算 ---
  const int tid = threadIdx.x;
  const int threads_per_block = blockDim.x;

  // Warp ID in the block
  int warpId = tid / warpSize;

  // Warp's position in the block tile
  int warp_m = warpId / WARPS_PER_BLOCK_N; // 0..3
  int warp_n = warpId % WARPS_PER_BLOCK_N; // 0..3

  // --- 声明WMMA片段 ---
  // A, B use half precision, C/D use float
  fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
  fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
  fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

  // 用 0 初始化累加器片段
  fill_fragment(acc_frag, 0.0f);

  // --- 共享内存 ---
  // Shared memory to stage tiles of A and B
  __shared__ half sA[BLOCK_TILE_M][BLOCK_TILE_K]; // 64 x 16
  __shared__ half sB[BLOCK_TILE_K][BLOCK_TILE_N]; // 16 x 64

  int tile_a_row = blockIdx.y * BLOCK_TILE_M;
  int tile_b_col = blockIdx.x * BLOCK_TILE_N;

  const half *A_half = reinterpret_cast<const half *>(A.elements);
  const half *B_half = reinterpret_cast<const half *>(B.elements);

  int numTiles = A.K / BLOCK_TILE_K;

  for (int i = 0; i < numTiles; ++i) {
    // --- 协作加载数据到共享内存 ---
    // 每个线程块加载一个 64x16 的 A tile 和一个 16x64 的 B tile
    // 每个线程加载 (64*16 + 16*64) / (128 threads) = 16 elem
    int gA_k = i * BLOCK_TILE_K;
    int gB_k = i * BLOCK_TILE_K;

    // --- load sA (BLOCK_TILE_M x BLOCK_TILE_K) as linear array ---
    const int elemsA = BLOCK_TILE_M * BLOCK_TILE_K; // 64*16 = 1024
    for (int idx = tid; idx < elemsA; idx += threads_per_block) {
      int r = idx / BLOCK_TILE_K; // 0..63
      int c = idx % BLOCK_TILE_K; // 0..15
      int g_r = tile_a_row + r;
      int g_c = gA_k + c;
      if (g_r < A.M && g_c < A.K) {
        sA[r][c] = A_half[g_r * A.K + g_c];
      } else {
        sA[r][c] = __float2half(0.0f);
      }
    }

    // --- load sB (BLOCK_TILE_K x BLOCK_TILE_N) as linear array ---
    const int elemsB = BLOCK_TILE_K * BLOCK_TILE_N; // 16*64 = 1024
    for (int idx = tid; idx < elemsB; idx += threads_per_block) {
      int r = idx / BLOCK_TILE_N; // 0..15
      int c = idx % BLOCK_TILE_N; // 0..63
      int g_r = gB_k + r;         // k index
      int g_c = tile_b_col + c;   // n index
      if (g_r < B.K && g_c < B.N) {
        sB[r][c] = B_half[g_r * B.N + g_c];
      } else {
        sB[r][c] = __float2half(0.0f);
      }
    }

    __syncthreads();

    // --- Warp级计算 ---
    // 每个 warp 计算一个 16x16 的 C tile
    // --- WMMA compute: break BLOCK_TILE_K into WMMA_K chunks (here
    // BLOCK_TILE_K==WMMA_K) --- For generality, loop k over BLOCK_TILE_K /
    // WMMA_K
    const int ksteps = BLOCK_TILE_K / WMMA_K; // should be 1 when both 16
    for (int ks = 0; ks < ksteps; ++ks) {
      // compute coordinates for fragment load
      int a_row = warp_m * WMMA_M; // row offset in sA
      int a_col = ks * WMMA_K;     // col offset in sA
      int b_row = ks * WMMA_K;     // row offset in sB
      int b_col = warp_n * WMMA_N; // col offset in sB

      // NOTE: lda for sA is BLOCK_TILE_K (number of columns in sA)
      load_matrix_sync(a_frag, &sA[a_row][a_col], BLOCK_TILE_K);
      // NOTE: lda for sB is BLOCK_TILE_N (number of columns in sB)
      load_matrix_sync(b_frag, &sB[b_row][b_col], BLOCK_TILE_N);

      mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    __syncthreads();
  } // for t

  // --- 写回结果 ---
  // 将累加器片段写回到C矩阵
  int c_m = blockIdx.y * BLOCK_TILE_M + warp_m * WMMA_M;
  int c_n = blockIdx.x * BLOCK_TILE_N + warp_n * WMMA_N;

  // C是FP32，所以不需要转换
  // boundary guard: if tile goes outside matrix, store only valid parts
  // WMMA store writes full 16x16; guard by checking starting indices
  if (c_m < C.M && c_n < C.N) {
    // if C.N is the ld
    store_matrix_sync(&C.elements[c_m * C.N + c_n], acc_frag, C.N,
                      mem_row_major);
  }
}

int main() {
  //--- 1. 初始化 ---
  // 1024x1024 矩阵
  const int M = 1 << 10;
  const int K = 1 << 10;
  const int N = 1 << 10;

#if defined(OP6)
  // A和B是 __half, C是 float
  auto aBytes = (size_t)M * K * sizeof(__half);
  auto bBytes = (size_t)K * N * sizeof(__half);
  auto cBytes = (size_t)M * N * sizeof(float);

  // 1. 创建临时的 float host 内存来初始化数据
  float *h_A_float = (float *)malloc(M * K * sizeof(float));
  float *h_B_float = (float *)malloc(K * N * sizeof(float));
  for (int i = 0; i < M * K; ++i)
    h_A_float[i] = 1.0f;
  for (int i = 0; i < K * N; ++i)
    h_B_float[i] = 2.0f;

  // 2. 创建 __half host 内存
  __half *h_A_half = (__half *)malloc(aBytes);
  __half *h_B_half = (__half *)malloc(bBytes);

  // 3. 执行从 float 到 half 的转换
  for (int i = 0; i < M * K; ++i)
    h_A_half[i] = __float2half(h_A_float[i]);
  for (int i = 0; i < K * N; ++i)
    h_B_half[i] = __float2half(h_B_float[i]);

  // 释放临时的float内存
  free(h_A_float);
  free(h_B_float);

#else // 其他优化等级的内存分配
  auto aBytes = (size_t)M * K * sizeof(float);
  auto bBytes = (size_t)K * N * sizeof(float);
  auto cBytes = (size_t)M * N * sizeof(float);
#endif

  Matrix h_A(M, K, N);
  Matrix h_B(M, K, N);
  Matrix h_C(M, K, N);

#if !defined(OP6)
  h_A.elements = (float *)malloc(aBytes);
  h_B.elements = (float *)malloc(bBytes);
  for (int i = 0; i < M * K; ++i)
    h_A.elements[i] = 1.0f;
  for (int i = 0; i < K * N; ++i)
    h_B.elements[i] = 2.0f;
#endif

  // 在 Device 上分配矩阵数据内存
  h_C.elements = (float *)malloc(cBytes);

  // Copy matrix meta
  Matrix d_A(h_A), d_B(h_B), d_C(h_C);

  // 为 device 矩阵分配内存
  CUDA_CHECK(cudaMalloc(&d_A.elements, aBytes));
  CUDA_CHECK(cudaMalloc(&d_B.elements, bBytes));
  CUDA_CHECK(cudaMalloc(&d_C.elements, cBytes));

  // 将 Host 数据拷贝到 Device
#if defined(OP6)
  CUDA_CHECK(
      cudaMemcpy(d_A.elements, h_A_half, aBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B.elements, h_B_half, bBytes, cudaMemcpyHostToDevice));
#else
  CUDA_CHECK(
      cudaMemcpy(d_A.elements, h_A.elements, aBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B.elements, h_B.elements, bBytes, cudaMemcpyHostToDevice));
#endif

  //--- 2. Kernel 执行与计时 ---
  dim3 blockSize;
  dim3 gridSize;

#if defined(OP0) || defined(OP1)
  blockSize = dim3(TILE_DIM, TILE_DIM);                    // 32, 32
  gridSize = dim3((h_C.N + blockSize.x - 1) / blockSize.x, // 32, 32
                  (h_C.M + blockSize.y - 1) / blockSize.y);
#elif defined(OP2) || defined(OP3) || defined(OP4) || defined(OP5)
  blockSize = dim3(BLOCK_COLS, BLOCK_ROWS);          // 8, 4
  gridSize = dim3((h_C.N + TILE_DIM - 1) / TILE_DIM, // 32, 32
                  (h_C.M + TILE_DIM - 1) / TILE_DIM);
#elif defined(OP6)
// Block tile size is 64x64
#define BLOCK_TILE_M 64
#define BLOCK_TILE_N 64
#define WARPS_PER_BLOCK 16 // 16 warps per block
#define THREADS_PER_WARP 32
  blockSize = dim3(WARPS_PER_BLOCK * THREADS_PER_WARP); // 512个线程
  gridSize = dim3((h_C.N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
                  (h_C.M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);
#endif
  // 创建 CUDA 事件用于计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 记录开始时间
  CUDA_CHECK(cudaEventRecord(start));

  // 执行 Kernel
  for (int i = 0; i < ITER; ++i) {
#if defined(OP0)
    mmKnlL0<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP1)
    mmKnlL1<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP2)
    mmKnlL2<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP3)
    mmKnlL3<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP4)
    mmKnlL4<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP5)
    mmKnlL5<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#elif defined(OP6)
    mmKnlL6<<<gridSize, blockSize>>>(d_A, d_B, d_C);
#else
    std::cout << "[Error] Optimize Level not targeted or defined." << std::endl;
    exit(0);
#endif
  }
  // 记录结束时间
  CUDA_CHECK(cudaEventRecord(stop));

  // 等待 Kernel 执行完毕
  CUDA_CHECK(cudaEventSynchronize(stop));

  // 计算执行时间
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Total iteration:" << ITER << "\n"
            << "Tatal Kernel Execution Time:" << milliseconds << "ms\n"
            << "Average Kernel Execution Time:" << milliseconds / ITER << "ms\n"
            << std::endl;

  //--- 3. 结果验证 ---
  CUDA_CHECK(
      cudaMemcpy(h_C.elements, d_C.elements, cBytes, cudaMemcpyDeviceToHost));

  int errors = 0;
  float maxError = 0.0f;
  for (int i = 0; i < N * N; ++i) {
    float fab = fabs(h_C.elements[i] - (2.0f * N));
    if (fab >
#if defined(OP5)
        0.01f
#else
        0.00001f
#endif
    )
      ++errors;
    maxError = fmax(maxError, fab);
  }
  if (!errors)
    std::cout << "Test PASS!" << std::endl;
  else
    std::cout << "Errors:" << errors << ", Max Error:" << maxError << std::endl;

    //--- 4. 资源释放 ---
#if defined(OP6)
  free(h_A_half);
  free(h_B_half);
#else
  free(h_A.elements);
  free(h_B.elements);
#endif
  free(h_C.elements);
  CUDA_CHECK(cudaFree(d_A.elements));
  CUDA_CHECK(cudaFree(d_B.elements));
  CUDA_CHECK(cudaFree(d_C.elements));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}